import hashlib
import re
import orjson
import tiktoken
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, RecursiveJsonSplitter
from semantic_text_splitter import CodeSplitter as RustASTCodeSplitter
from tree_sitter_language_pack import get_parser, get_language
from services.chunking.base import BaseChunkingStrategy
from models.schemas import ChunkDocument, ChunkMetadata

# Rust-backed token counter
_ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens_rust(text: str) -> int:
    return len(_ENCODER.encode(text))

def extract_structural_headers(content: str, ext: str) -> str:
    """Extracts top-level imports, package declarations, or class/function signatures for relational context."""
    lines = content.splitlines()[:50]
    headers = []
    for line in lines:
        stripped = line.strip()
        if ext == ".py" and (stripped.startswith("import ") or stripped.startswith("from ") or stripped.startswith("class ") or stripped.startswith("def ")):
            headers.append(stripped)
        elif ext in (".js", ".tsx", ".ts") and (stripped.startswith("import ") or stripped.startswith("export ") or "function " in stripped or "class " in stripped):
            headers.append(stripped)
        elif ext == ".java" and (stripped.startswith("package ") or stripped.startswith("import ") or "class " in stripped or "interface " in stripped):
            headers.append(stripped)
    return " | ".join(headers[:8])

def build_linked_chunks(
    raw_texts: List[str],
    repo_id: str,
    file_path: str,
    file_hash: str,
    language: str,
    strategy_name: str,
    structural_context: str
) -> List[ChunkDocument]:
    """Creates doubly-linked ChunkDocuments with shared sibling_group_id for relational retrieval."""
    total = len(raw_texts)
    sibling_group_id = hashlib.sha1(f"{repo_id}:{file_path}".encode("utf-8")).hexdigest()[:16]
    chunk_ids = [
        hashlib.sha1(f"{repo_id}:{file_path}:{idx}:{file_hash}".encode("utf-8")).hexdigest()[:20]
        for idx in range(total)
    ]
    
    docs: List[ChunkDocument] = []
    for idx, text in enumerate(raw_texts):
        prev_id = chunk_ids[idx - 1] if idx > 0 else None
        next_id = chunk_ids[idx + 1] if idx < total - 1 else None
        
        meta = ChunkMetadata(
            chunk_id=chunk_ids[idx],
            repo_id=repo_id,
            file_path=file_path,
            file_hash=file_hash,
            language=language,
            strategy_used=strategy_name,
            chunk_index=idx,
            total_chunks_in_file=total,
            prev_chunk_id=prev_id,
            next_chunk_id=next_id,
            sibling_group_id=sibling_group_id,
            structural_context=structural_context,
            token_count=count_tokens_rust(text)
        )
        enriched_content = f"// File: {file_path}\n"
        if structural_context:
            enriched_content += f"// Context: {structural_context}\n"
        enriched_content += f"\n{text}"
        
        docs.append(ChunkDocument(chunk_id=chunk_ids[idx], content=enriched_content, metadata=meta))
    return docs

class ASTLanguageChunkStrategy(BaseChunkingStrategy):
    """
    True Abstract Syntax Tree (AST) chunker powered by Tree-sitter & Rust's CodeSplitter.
    """
    
    EXT_TO_TS_LANG = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".java": "java",
        ".css": "css",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
    }

    DEFINITION_NODE_TYPES = {
        "class_definition", "function_definition", "decorated_definition",
        "class_declaration", "function_declaration", "method_definition",
        "lexical_declaration", "export_statement",
        "interface_declaration", "type_alias_declaration",
        "method_declaration", "constructor_declaration",
        "rule_set", "media_statement",
        "block_mapping_pair",
    }

    IMPORT_NODE_TYPES = {
        "import_statement", "import_from_statement",
        "import_declaration", "package_declaration",
    }

    def __init__(self, ext: str):
        self.ext = ext.lower()
        self.ts_lang_name = self.EXT_TO_TS_LANG.get(self.ext, "python")

    def _get_node_symbol_name(self, node, source_bytes: bytes) -> str:
        name_node = node.child_by_field_name("name")
        if name_node:
            return source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
        return node.type

    def _extract_ast_units(self, root_node, source_bytes: bytes, chunk_size: int):
        imports = []
        ast_units = []

        def walk(node, scope_chain: List[str]):
            for child in node.children:
                if child.type in self.IMPORT_NODE_TYPES:
                    imp_text = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                    imports.append(imp_text)
                    continue

                if child.type in self.DEFINITION_NODE_TYPES:
                    symbol = self._get_node_symbol_name(child, source_bytes)
                    current_scope = scope_chain + [f"{child.type}:{symbol}"]
                    node_bytes = source_bytes[child.start_byte:child.end_byte]
                    node_text = node_bytes.decode("utf-8", errors="replace")

                    if len(node_text) <= chunk_size:
                        ast_units.append((child.start_byte, child.end_byte, node_text, " -> ".join(current_scope)))
                    else:
                        before_count = len(ast_units)
                        walk(child, current_scope)
                        if len(ast_units) == before_count:
                            try:
                                lang_pointer = get_language(self.ts_lang_name)
                                rust_ast_splitter = RustASTCodeSplitter(lang_pointer, chunk_size)
                                sub_chunks = list(rust_ast_splitter.chunks(node_text))
                            except Exception:
                                sub_chunks = [node_text[i:i+chunk_size] for i in range(0, len(node_text), chunk_size)]
                            for part_idx, sub in enumerate(sub_chunks):
                                ast_units.append((child.start_byte, child.end_byte, sub, f"{' -> '.join(current_scope)} [part {part_idx+1}]"))
                else:
                    walk(child, scope_chain)

        walk(root_node, [])

        # Gap-fill: keep everything the walker skipped (top-level statements,
        # comments, imports) so no source code is silently dropped from the index.
        def _split_by_lines(text: str) -> List[str]:
            parts, buf = [], ""
            for ln in text.splitlines(keepends=True):
                if len(buf) + len(ln) > chunk_size and buf:
                    parts.append(buf)
                    buf = ""
                buf += ln
            if buf:
                parts.append(buf)
            return parts

        ast_units.sort(key=lambda u: u[0])  # stable: sub-parts keep their order
        filled, pos = [], 0
        for start, end, unit_text, scope in ast_units:
            if start > pos:
                gap = source_bytes[pos:start].decode("utf-8", errors="replace").strip()
                if gap:
                    filled.extend((part, "top_level_code") for part in _split_by_lines(gap))
            filled.append((unit_text, scope))
            pos = max(pos, end)
        tail = source_bytes[pos:].decode("utf-8", errors="replace").strip()
        if tail:
            filled.extend((part, "top_level_code") for part in _split_by_lines(tail))
        return imports, filled

    def chunk_file(
        self,
        repo_id: str,
        file_path: str,
        file_hash: str,
        content: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[ChunkDocument]:
        source_bytes = content.encode("utf-8")
        parser = get_parser(self.ts_lang_name)
        tree = parser.parse(source_bytes)

        imports, ast_units = self._extract_ast_units(tree.root_node, source_bytes, chunk_size)
        import_header = " | ".join(imports[:8])

        if not ast_units:
            try:
                lang_pointer = get_language(self.ts_lang_name)
                rust_ast_splitter = RustASTCodeSplitter(lang_pointer, chunk_size)
                raw_chunks = list(rust_ast_splitter.chunks(content)) or [content]
            except Exception:
                raw_chunks = [content]
            return build_linked_chunks(
                raw_chunks, repo_id, file_path, file_hash,
                self.ts_lang_name, f"TreeSitterRustAST({self.ts_lang_name})", import_header
            )

        merged_texts: List[str] = []
        merged_scopes: List[str] = []
        current_buf: List[str] = []
        current_scopes: List[str] = []
        current_len = 0

        for unit_text, scope_str in ast_units:
            if current_len + len(unit_text) > chunk_size and current_buf:
                merged_texts.append("\n\n".join(current_buf))
                merged_scopes.append(" ; ".join(current_scopes))
                current_buf, current_scopes, current_len = [], [], 0
            current_buf.append(unit_text)
            current_scopes.append(scope_str)
            current_len += len(unit_text)

        if current_buf:
            merged_texts.append("\n\n".join(current_buf))
            merged_scopes.append(" ; ".join(current_scopes))

        docs = build_linked_chunks(
            merged_texts, repo_id, file_path, file_hash,
            self.ts_lang_name, f"TreeSitterAST({self.ts_lang_name})", import_header
        )
        for doc, scope_breadcrumb in zip(docs, merged_scopes):
            combined_ctx = f"AST_Scope: [{scope_breadcrumb}]"
            if import_header:
                combined_ctx += f" | Imports: {import_header}"
            doc.metadata.structural_context = combined_ctx
            doc.content = f"// File: {file_path}\n// {combined_ctx}\n\n{doc.content.split(chr(10)+chr(10), 1)[-1]}"

        return docs

class StylesheetAndConfigStrategy(BaseChunkingStrategy):
    """Handles .css, .yaml, .yml using native Rust semantic-text-splitter."""
    
    def __init__(self, ext: str):
        self.ext = ext

    def chunk_file(self, repo_id: str, file_path: str, file_hash: str, content: str, chunk_size: int, chunk_overlap: int) -> List[ChunkDocument]:
        from semantic_text_splitter import TextSplitter as RustTextSplitter
        rust_splitter = RustTextSplitter(chunk_size, chunk_overlap)
        raw_chunks = list(rust_splitter.chunks(content)) or [content]
        selectors = re.findall(r"^([a-zA-Z0-9_\.\-#@][^{:\n]+)[{:]", content, re.MULTILINE)[:8]
        context = " | ".join(s.strip() for s in selectors)
        return build_linked_chunks(
            raw_chunks, repo_id, file_path, file_hash,
            self.ext.lstrip("."), "RustSemanticSplitter", context
        )

class HtmlStrategy(BaseChunkingStrategy):
    """Handles .html/.htm with tag-aware splitting; lists <script src> tags in every chunk's context."""

    def chunk_file(self, repo_id: str, file_path: str, file_hash: str, content: str, chunk_size: int, chunk_overlap: int) -> List[ChunkDocument]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.HTML, chunk_size=chunk_size, chunk_overlap=0
        )
        raw_chunks = splitter.split_text(content) or [content]
        srcs = re.findall(r'<script[^>]*\ssrc=["\']([^"\']+)["\']', content)
        context = ("Scripts: " + ", ".join(srcs[:8])) if srcs else "html"
        return build_linked_chunks(
            raw_chunks, repo_id, file_path, file_hash,
            "html", "LangChainHTMLSplitter", context
        )

class JsonAndJsonlStrategy(BaseChunkingStrategy):
    """Handles .json and .jsonl using orjson and hierarchical splitters."""
    
    def __init__(self, ext: str):
        self.ext = ext

    def chunk_file(self, repo_id: str, file_path: str, file_hash: str, content: str, chunk_size: int, chunk_overlap: int) -> List[ChunkDocument]:
        raw_chunks: List[str] = []
        context = f"Format: {self.ext}"
        
        if self.ext == ".jsonl":
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            batch, current_len = [], 0
            for idx, line in enumerate(lines):
                try:
                    parsed = orjson.loads(line)
                    formatted = orjson.dumps(parsed).decode("utf-8")
                except Exception:
                    formatted = line
                if current_len + len(formatted) > chunk_size and batch:
                    raw_chunks.append("\n".join(batch))
                    batch, current_len = [], 0
                batch.append(f"[Record {idx}] {formatted}")
                current_len += len(formatted)
            if batch:
                raw_chunks.append("\n".join(batch))
        else:
            try:
                parsed_json = orjson.loads(content)
                if isinstance(parsed_json, dict):
                    context = "TopKeys: " + ", ".join(list(parsed_json.keys())[:10])
                    splitter = RecursiveJsonSplitter(max_chunk_size=chunk_size)
                    raw_chunks = splitter.split_text(json_data=parsed_json, convert_lists=True)
                else:
                    from semantic_text_splitter import TextSplitter as RustTextSplitter
                    rust_splitter = RustTextSplitter(chunk_size, chunk_overlap)
                    raw_chunks = list(rust_splitter.chunks(content))
            except Exception:
                from semantic_text_splitter import TextSplitter as RustTextSplitter
                rust_splitter = RustTextSplitter(chunk_size, chunk_overlap)
                raw_chunks = list(rust_splitter.chunks(content))

        if not raw_chunks:
            raw_chunks = [content]

        return build_linked_chunks(
            raw_chunks, repo_id, file_path, file_hash,
            self.ext.lstrip("."), "OrjsonHierarchicalSplitter", context
        )
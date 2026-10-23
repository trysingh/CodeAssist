import time
import sys
import os
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import settings
from models.schemas import ChunkDocument
from services.chunking.strategies import count_tokens_rust

class VectorStoreService:
    def __init__(self):
        # Force Ollama Python SDK to honor remote base URL
        os.environ["OLLAMA_HOST"] = settings.ollama_base_url
        
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url
        )

    def _get_vectorstore(self, repo_id: str) -> Chroma:
        collection_name = f"repo_{repo_id}"
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_path
        )

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def embed_and_persist_chunks(self, repo_id: str, chunks: List[ChunkDocument]) -> None:
        if not chunks:
            return
        start_t = time.perf_counter()
        vectorstore = self._get_vectorstore(repo_id)
        
        lc_docs: List[Document] = []
        ids: List[str] = []
        total_input_tokens = 0
        
        # Strict character capping to prevent runner context crashes
        MAX_SAFE_CHARS = 2000

        for ch in chunks:
            clean_content = ch.content.replace("\x00", "").strip()
            if not clean_content:
                continue
            if len(clean_content) > MAX_SAFE_CHARS:
                clean_content = clean_content[:MAX_SAFE_CHARS]

            total_input_tokens += count_tokens_rust(clean_content)
            meta_dict = {
                "chunk_id": ch.metadata.chunk_id,
                "repo_id": ch.metadata.repo_id,
                "file_path": ch.metadata.file_path,
                "language": ch.metadata.language,
                "strategy_used": ch.metadata.strategy_used,
                "chunk_index": ch.metadata.chunk_index,
                "prev_chunk_id": ch.metadata.prev_chunk_id or "",
                "next_chunk_id": ch.metadata.next_chunk_id or "",
                "sibling_group_id": ch.metadata.sibling_group_id,
                "structural_context": ch.metadata.structural_context[:300],
            }
            lc_docs.append(Document(page_content=clean_content, metadata=meta_dict))
            ids.append(ch.chunk_id)

        # Process in tiny sub-batches of 4 to prevent overloading remote Ollama runner
        SUB_BATCH = 4
        for i in range(0, len(lc_docs), SUB_BATCH):
            vectorstore.add_documents(
                documents=lc_docs[i : i + SUB_BATCH],
                ids=ids[i : i + SUB_BATCH]
            )

        elapsed = time.perf_counter() - start_t
        sys.stdout.write(
            f"[TELEMETRY TRACE] Phase=EMBED_BATCH | Model={settings.embedding_model} | "
            f"Chunks={len(lc_docs)} | InputTokens={total_input_tokens} | TimeSpent={elapsed:.3f}s\n"
        )
        sys.stdout.flush()

    def similarity_search(self, repo_id: str, query: str, k: int = 4) -> List[Document]:
        vectorstore = self._get_vectorstore(repo_id)
        return vectorstore.similarity_search(query, k=k)

    def delete_chunks_by_file(self, repo_id: str, file_path: str) -> None:
        """Removes all existing vector chunks associated with a specific file path to prevent duplicates."""
        try:
            vectorstore = self._get_vectorstore(repo_id)
            # Fetch existing chunk IDs for this file path using Chroma metadata filter
            existing = vectorstore.get(where={"file_path": file_path})
            if existing and existing.get("ids"):
                vectorstore.delete(ids=existing["ids"])
        except Exception as exc:
            sys.stdout.write(f"[WARNING] Failed to purge old vectors for {file_path}: {exc}\n")
            sys.stdout.flush()

    def delete_collection(self, repo_id: str) -> None:
        """Deletes the entire vector collection for a specific repository."""
        try:
            collection_name = f"repo_{repo_id}"
            vectorstore = self._get_vectorstore(repo_id)
            # Chroma's delete_collection is usually called on the client, 
            # but we can use the underlying client from the vectorstore object.
            vectorstore._client.delete_collection(collection_name)
        except Exception as exc:
            sys.stdout.write(f"[WARNING] Failed to delete collection for {repo_id}: {exc}\n")
            sys.stdout.flush()
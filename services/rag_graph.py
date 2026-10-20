from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from config.settings import settings
from models.database import DatabaseManager
from services.vector_store import VectorStoreService
from services.telemetry import StdoutTelemetryCallback
import re
MAX_FILE_CHARS = 70000

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    repo_id: str
    context_str: str

class RAGWorkflowManager:
    """
    LangGraph state machine with relational chunk expansion and SQLite conversation persistence.
    """
    def __init__(self, db_manager: DatabaseManager, vector_service: VectorStoreService):
        self.db_manager = db_manager
        self.vector_service = vector_service

    async def _retrieve_and_expand_node(self, state: AgentState) -> dict:
        repo_id = state["repo_id"]
        last_user_msg = state["messages"][-1].content

        query = str(last_user_msg).lower()
        all_paths = await self.db_manager.list_file_paths(repo_id)
        named = [p for p in all_paths if re.search(
            r'(?<![\w.\-])' + re.escape(p.replace("\\", "/").split("/")[-1].lower()) + r'(?![\w])', query)]
        if named:
            blocks, used = [], 0
            for p in sorted(named):
                chunks = await self.db_manager.get_file_chunks(repo_id, p)
                text = "\n\n".join(c["content"] for c in chunks)
                status = "COMPLETE"
                if used + len(text) > MAX_FILE_CHARS:
                    text, status = text[: max(0, MAX_FILE_CHARS - used)], "TRUNCATED (budget)"
                used += len(text)
                blocks.append(f"--- [FULL_FILE: {status}] {p} ({len(chunks)} chunks) ---\n{text}")
            return {"context_str": "\n\n".join(blocks)}



        
        # 1. Primary semantic search in ChromaDB
        primary_docs = self.vector_service.similarity_search(repo_id, str(last_user_msg), k=4)
        
        # 2. Collect adjacent chunk IDs (prev_chunk_id / next_chunk_id) to place related chunks together
        neighbor_ids = set()
        primary_ids = set()
        for doc in primary_docs:
            cid = doc.metadata.get("chunk_id")
            if cid:
                primary_ids.add(cid)
            prev_id = doc.metadata.get("prev_chunk_id")
            next_id = doc.metadata.get("next_chunk_id")
            if prev_id:
                neighbor_ids.add(prev_id)
            if next_id:
                neighbor_ids.add(next_id)
                
        missing_neighbor_ids = list(neighbor_ids - primary_ids)
        neighbor_records = await self.db_manager.get_adjacent_chunks(missing_neighbor_ids)
        
        # 3. Group and order chunks by file_path and chunk_index so related code is read contiguously
        grouped_chunks = []
        for doc in primary_docs:
            grouped_chunks.append({
                "file_path": doc.metadata.get("file_path", "unknown"),
                "chunk_index": int(doc.metadata.get("chunk_index", 0)),
                "content": doc.page_content,
                "role": "PRIMARY_MATCH"
            })
        for rec in neighbor_records:
            grouped_chunks.append({
                "file_path": rec["file_path"],
                "chunk_index": int(rec["chunk_index"]),
                "content": rec["content"],
                "role": "RELATIONAL_NEIGHBOR"
            })
            
        grouped_chunks.sort(key=lambda x: (x["file_path"], x["chunk_index"]))
        
        formatted_blocks = []
        for item in grouped_chunks:
            header = f"--- [{item['role']}] File: {item['file_path']} (Chunk #{item['chunk_index']}) ---"
            formatted_blocks.append(f"{header}\n{item['content']}")
            
        context_str = "\n\n".join(formatted_blocks) if formatted_blocks else "No relevant code chunks found."
        return {"context_str": context_str}

    async def _generate_answer_node(self, state: AgentState) -> dict:
        callback = StdoutTelemetryCallback(model_name=settings.llm_model, phase="RAG_GENERATION")
        llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
            num_ctx=32768,
            callbacks=[callback]
        )
        
        system_prompt = (
            "You are an expert software architect analyzing a code repository.\n"
            "Use the retrieved code chunks (which include relational neighbor chunks from the same files) "
            "to accurately answer the user's question. "
            "Blocks marked FULL_FILE: COMPLETE are entire files. Otherwise the file is partial, so never conclude something is absent. "
            "Always cite specific file paths and functions.\n\n"
            f"=== RETRIEVED REPOSITORY CONTEXT ===\n{state.get('context_str', '')}\n===================================="
        )
        
        messages_for_llm = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await llm.ainvoke(messages_for_llm)
        return {"messages": [response]}

    def build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("retrieve_and_expand", self._retrieve_and_expand_node)
        builder.add_node("generate", self._generate_answer_node)
        builder.add_edge(START, "retrieve_and_expand")
        builder.add_edge("retrieve_and_expand", "generate")
        builder.add_edge("generate", END)
        return builder

    async def invoke_chat(self, repo_id: str, thread_id: str, user_input: str) -> str:
        builder = self.build_graph()
        async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_path) as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": f"{repo_id}:{thread_id}"}}
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=user_input)], "repo_id": repo_id},
                config=config
            )
            return result["messages"][-1].content

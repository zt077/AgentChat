from typing import Optional

from loguru import logger

from agentchat.schema.knowledge import KnowledgeContextChunk, KnowledgeContextPackage
from agentchat.services.rag.es_client import client as es_client
from agentchat.services.rag.rerank import Reranker
from agentchat.services.rag.retrieval import MixRetrival
from agentchat.services.rag.vector_db import milvus_client
from agentchat.services.rewrite.query_write import query_rewriter
from agentchat.settings import app_settings


class RagHandler:
    @staticmethod
    def _build_excerpt(content: str, max_length: int = 450) -> str:
        if len(content) <= max_length:
            return content
        return f"{content[: max_length - 3]}..."

    @classmethod
    async def query_rewrite(cls, query):
        return await query_rewriter.rewrite(query)

    @classmethod
    async def index_milvus_documents(cls, collection_name, chunks):
        await milvus_client.insert(collection_name, chunks)

    @classmethod
    async def index_es_documents(cls, index_name, chunks):
        await es_client.index_documents(index_name, chunks)

    @classmethod
    async def mix_retrival_documents(cls, query_list, knowledges_id, search_field="summary"):
        if app_settings.rag.enable_elasticsearch:
            es_documents, milvus_documents = await MixRetrival.mix_retrival_documents(query_list, knowledges_id, search_field)
            es_documents.sort(key=lambda x: x.score, reverse=True)
            milvus_documents.sort(key=lambda x: x.score, reverse=True)
            all_documents = es_documents + milvus_documents
        else:
            all_documents = await MixRetrival.retrival_milvus_documents(query_list, knowledges_id, search_field)

        documents = []
        seen_chunk_ids = set()
        all_documents.sort(key=lambda x: x.score, reverse=True)

        for doc in all_documents:
            if doc.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(doc.chunk_id)
                documents.append(doc)
                if len(documents) >= 10:
                    break

        return documents

    @classmethod
    async def build_context_package(
        cls,
        query,
        knowledges_id,
        min_score: Optional[float] = None,
        top_k: Optional[int] = None,
        needs_query_rewrite: bool = True,
    ) -> KnowledgeContextPackage:
        if min_score is None:
            min_score = app_settings.rag.retrival.get("min_score")
        if top_k is None:
            top_k = app_settings.rag.retrival.get("top_k")

        rewritten_queries = await cls.query_rewrite(query) if needs_query_rewrite else [query]
        retrieved_documents = await cls.mix_retrival_documents(rewritten_queries, knowledges_id, "content")
        if not retrieved_documents:
            return KnowledgeContextPackage(query=query, rewritten_queries=rewritten_queries)

        reranked_docs = await Reranker.rerank_documents(query, [doc.content for doc in retrieved_documents])
        selected_items = []
        actual_top_k = top_k if top_k is not None else len(reranked_docs)
        for reranked_doc in reranked_docs[:actual_top_k]:
            if min_score is not None and reranked_doc.score < min_score:
                continue
            selected_items.append((retrieved_documents[reranked_doc.index], reranked_doc.score))

        if not selected_items:
            return KnowledgeContextPackage(query=query, rewritten_queries=rewritten_queries)

        retrieval_strategy = "summary-first" if len(query) <= 16 else "content-focused"
        context_blocks = []
        citations = []
        compact_parts = []

        for idx, (doc, score) in enumerate(selected_items, start=1):
            file_name = getattr(doc, "file_name", "") or getattr(doc, "knowledge_id", f"source_{idx}")
            citation = f"[{idx}] {file_name}"
            excerpt = cls._build_excerpt(getattr(doc, "content", ""))
            context_blocks.append(
                KnowledgeContextChunk(
                    chunk_id=getattr(doc, "chunk_id", ""),
                    knowledge_id=getattr(doc, "knowledge_id", ""),
                    file_id=getattr(doc, "file_id", ""),
                    file_name=file_name,
                    score=score,
                    excerpt=excerpt,
                )
            )
            citations.append(citation)
            compact_parts.append(f"{citation} score={score:.3f}\n{excerpt}")

        return KnowledgeContextPackage(
            query=query,
            rewritten_queries=rewritten_queries,
            retrieval_strategy=retrieval_strategy,
            context_blocks=context_blocks,
            compact_context="\n\n".join(compact_parts),
            citations=citations,
        )

    @classmethod
    async def rag_query_summary(
        cls,
        query,
        knowledges_id,
        min_score: Optional[float] = None,
        top_k: Optional[int] = None,
        needs_query_rewrite: bool = True,
    ):
        context_package = await cls.build_context_package(
            query=query,
            knowledges_id=knowledges_id,
            min_score=min_score,
            top_k=top_k,
            needs_query_rewrite=needs_query_rewrite,
        )
        if context_package.compact_context:
            return context_package.compact_context
        logger.info("Summary retrieval was insufficient, falling back to ranked content retrieval")
        return await cls.retrieve_ranked_documents(query, knowledges_id, knowledges_id)

    @classmethod
    async def retrieve_ranked_documents(
        cls,
        query,
        collection_names,
        index_names=None,
        min_score: Optional[float] = None,
        top_k: Optional[int] = None,
        needs_query_rewrite: bool = True,
    ):
        context_package = await cls.build_context_package(
            query=query,
            knowledges_id=collection_names,
            min_score=min_score,
            top_k=top_k,
            needs_query_rewrite=needs_query_rewrite,
        )
        return context_package.compact_context or "No relevant documents found."

    @classmethod
    async def delete_documents_es_milvus(cls, file_id, knowledge_id):
        if app_settings.rag.enable_elasticsearch:
            await es_client.delete_documents(file_id, knowledge_id)
        await milvus_client.delete_by_file_id(file_id, knowledge_id)

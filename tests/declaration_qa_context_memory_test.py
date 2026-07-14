from declaration_qa.application.factory import create_workflow
from declaration_qa.application.agent import AgentDecision
from declaration_qa.api.schemas import ChatRequest
from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.memory.manager import ContextMemoryManager
from declaration_qa.memory.retrieval import ContextMemoryRetriever
from declaration_qa.memory.store import InMemoryContextMemoryStore
from declaration_qa.rag.model_providers.embeddings.hash_embedding_provider import HashEmbeddingProvider
from declaration_qa.search.vector.memory import MemoryVectorIndex
from declaration_qa.storage.cache import InMemoryCacheStore


def test_short_memory_uses_cache_for_agent_messages_only():
    manager = build_manager()

    first = manager.load_short_memory(user_id=1, session_id="s1")
    manager.update_short_memory(
        payload={
            "user_id": 1,
            "session_id": "s1",
            "agent_messages": [
                {"role": "user", "content": "取水许可怎么办理"},
                {"role": "assistant", "content": "请按办事指南办理。"},
            ],
        }
    )
    second = manager.load_short_memory(user_id=1, session_id="s1")

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["agent_messages"][0]["content"] == "取水许可怎么办理"
    assert set(second) == {"version", "agent_messages", "cache_hit", "backend"}


def test_tool_messages_enter_short_memory_without_truncating_results():
    manager = build_manager()
    payload = {
        "user_id": 10,
        "session_id": "s10",
        "agent_messages": [
            {"role": "user", "content": "取水许可怎么办理"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "name": "knowledge_search"}]},
            {"role": "tool", "tool_call_id": "call-1", "name": "knowledge_search", "content": {"data": {"sources": [{"chunk_id": "chunk-water-process"}]}}},
        ],
    }

    manager.update_short_memory(payload=payload)
    cached = manager.load_short_memory(user_id=10, session_id="s10")

    assert cached["agent_messages"][2]["content"]["data"]["sources"][0]["chunk_id"] == "chunk-water-process"


def test_old_short_memory_cache_is_migrated_to_agent_messages_only():
    manager = build_manager()
    key = "declaration_qa:context:short:11:s11"
    manager.cache_store.set_json(
        key,
        {
            "version": 2,
            "recent_messages": [],
            "verified_facts": [{"reason": "旧事实"}],
            "ticket_item_overviews": [{"item_code": "ITEM-OLD"}],
            "agent_messages": [{"role": "user", "content": "继续"}],
        },
        ttl_seconds=60,
    )

    migrated = manager.load_short_memory(user_id=11, session_id="s11")

    assert migrated["version"] == 4
    assert migrated["agent_messages"] == [{"role": "user", "content": "继续"}]
    assert set(migrated) == {"version", "agent_messages", "cache_hit", "backend"}


def test_long_memory_async_indexes_and_retrieves_hybrid_context():
    manager = build_manager(distance_threshold=1.0, keyword_min_score=0.0)
    payload = {
        "user_id": 7,
        "session_id": "s7",
        "query": "我刚才问的取水许可申请材料是什么",
        "answer": "取水许可申请材料包括取水许可申请书、水资源论证报告书、利害关系说明等。",
        "intent": "material",
        "route": "ticket_context_unrelated",
        "region_code": "340000",
        "retrieved_sources": [
            {
                "title": "item_28.md",
                "metadata": {
                    "item_name": "取水许可",
                    "section_path": "取水许可办事指南 > 申请材料",
                },
            }
        ],
    }

    status = manager.after_interaction(payload, audit_id="audit-1")
    results = manager.wait_for_pending()
    context = manager.retrieve_long_term_context({"user_id": 7, "query": "刚才那个取水许可要哪些材料"})

    assert status["long_memory"]["async_enqueued"] is True
    assert results[0]["memory_count"] == 1
    assert context["items"]
    assert "取水许可申请材料" in context["items"][0]["content"]
    assert context["status"]["channels"]["dense"] + context["status"]["channels"]["bm25"] >= 1


def test_long_memory_distance_threshold_filters_dense_far_matches():
    manager = build_manager(distance_threshold=0.0, keyword_min_score=1.1)
    payload = {
        "user_id": 8,
        "session_id": "s8",
        "query": "草原防火许可材料",
        "answer": "草原防火许可需要申请表等材料。",
        "intent": "material",
        "retrieved_sources": [],
    }

    manager.after_interaction(payload, audit_id="audit-2")
    manager.wait_for_pending()
    context = manager.retrieve_long_term_context({"user_id": 8, "query": "zzzz yyyy xxxx"})

    assert context["items"] == []
    assert context["status"]["distance_threshold"] == 0.0


def test_factory_wires_context_memory_manager():
    workflow = create_workflow(
        DeclarationQASettings(
            _env_file=None,
            DECLARATION_QA_SEED_KNOWLEDGE=False,
            DECLARATION_QA_CONTEXT_MEMORY_ENABLED=True,
            DECLARATION_QA_EMBEDDING_PROVIDER="hashing",
            DECLARATION_QA_EMBEDDING_DIMENSIONS=64,
            DECLARATION_QA_VECTOR_BACKEND="memory",
            DECLARATION_QA_KEYWORD_BACKEND="memory",
            DECLARATION_QA_LLM_PROVIDER="deterministic",
        )
    )

    assert workflow.context_memory_manager is not None
    assert workflow.context_memory_manager.enabled is True


def test_factory_can_disable_context_memory_without_external_memory_indexes():
    workflow = create_workflow(
        DeclarationQASettings(
            _env_file=None,
            DECLARATION_QA_SEED_KNOWLEDGE=False,
            DECLARATION_QA_CONTEXT_MEMORY_ENABLED=False,
            DECLARATION_QA_EMBEDDING_PROVIDER="hashing",
            DECLARATION_QA_EMBEDDING_DIMENSIONS=64,
            DECLARATION_QA_VECTOR_BACKEND="memory",
            DECLARATION_QA_KEYWORD_BACKEND="memory",
            DECLARATION_QA_LLM_PROVIDER="deterministic",
        )
    )

    assert workflow.context_memory_manager is not None
    assert workflow.context_memory_manager.enabled is False


def test_factory_uses_redis_cache_store_for_context_memory(monkeypatch):
    import declaration_qa.application.factory as factory

    class FakeRedisCacheStore(InMemoryCacheStore):
        backend = "redis"

        def __init__(self, redis_url: str, *, allow_fallback: bool = True):
            super().__init__()
            self.redis_url = redis_url
            self.allow_fallback = allow_fallback

    monkeypatch.setattr(factory, "RedisCacheStore", FakeRedisCacheStore)
    workflow = create_workflow(
        DeclarationQASettings(
            _env_file=None,
            DECLARATION_QA_SEED_KNOWLEDGE=False,
            DECLARATION_QA_CONTEXT_MEMORY_ENABLED=True,
            DECLARATION_QA_REDIS_URL="redis://localhost:6380/1",
            DECLARATION_QA_EMBEDDING_PROVIDER="hashing",
            DECLARATION_QA_EMBEDDING_DIMENSIONS=64,
            DECLARATION_QA_VECTOR_BACKEND="memory",
            DECLARATION_QA_KEYWORD_BACKEND="memory",
            DECLARATION_QA_LLM_PROVIDER="deterministic",
        )
    )

    assert workflow.context_memory_manager.cache_store.backend == "redis"
    assert workflow.context_memory_manager.cache_store.redis_url == "redis://localhost:6380/1"


def test_workflow_async_memory_is_available_to_followup_prompt():
    workflow = create_workflow(
        DeclarationQASettings(
            _env_file=None,
            DECLARATION_QA_SEED_KNOWLEDGE=True,
            DECLARATION_QA_CONTEXT_MEMORY_ENABLED=True,
            DECLARATION_QA_EMBEDDING_PROVIDER="hashing",
            DECLARATION_QA_EMBEDDING_DIMENSIONS=64,
            DECLARATION_QA_VECTOR_BACKEND="memory",
            DECLARATION_QA_KEYWORD_BACKEND="memory",
            DECLARATION_QA_LLM_PROVIDER="deterministic",
        )
    )

    class KnowledgeCallingAgent:
        def decide(self, state, tool_definitions):
            if not tool_definitions or state.get("tool_results"):
                return AgentDecision("finish", final_answer="已根据现有证据完成回答。")
            return AgentDecision(
                "tools",
                tool_calls=[
                    {
                        "id": "knowledge",
                        "name": "knowledge_search",
                        "arguments": {
                            "queries": [
                                {
                                    "text": state.get("normalized_query") or state.get("query") or "",
                                    "strategy": "hybrid",
                                }
                            ]
                        },
                    }
                ],
            )

    workflow.agent = KnowledgeCallingAgent()

    first = workflow.invoke(ChatRequest(query="办理居住证需要哪些材料", user_id=21, region_code="340000"))
    memory_results = workflow.context_memory_manager.wait_for_pending()
    second = workflow.invoke(
        ChatRequest(
            query="刚才我问的材料再说一遍",
            user_id=21,
            region_code="340000",
            session_id=first.session_id,
        )
    )
    audit = workflow.audit_store.get(second.audit_id)

    assert memory_results[0]["memory_count"] == 1
    assert audit["long_term_context"]["items"]
    memory_content = " ".join(item["content"] for item in audit["long_term_context"]["items"])
    assert "办理居住证需要哪些材料" in memory_content


def build_manager(*, distance_threshold: float = 0.4, keyword_min_score: float = 0.2) -> ContextMemoryManager:
    store = InMemoryContextMemoryStore()
    embedding_provider = HashEmbeddingProvider(dimension=64)
    return ContextMemoryManager(
        cache_store=InMemoryCacheStore(),
        memory_store=store,
        retriever=ContextMemoryRetriever(
            store=store,
            vector_index=MemoryVectorIndex(),
            embedding_provider=embedding_provider,
            keyword_index=None,
            distance_threshold=distance_threshold,
            keyword_min_score=keyword_min_score,
        ),
        async_workers=1,
        short_ttl_seconds=60,
        long_ttl_seconds=600,
        retrieval_cache_ttl_seconds=60,
    )

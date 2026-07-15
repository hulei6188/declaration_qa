from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from declaration_qa.api.schemas import ChatRequest
from declaration_qa.application.agent import AgentDecision
from declaration_qa.application.workflow import DeclarationQAWorkflow
from declaration_qa.memory.runtime import (
    LangGraphMemoryRuntime,
    build_memory_resources,
    message_dicts_to_langchain,
    psycopg_connection_url,
)
from declaration_qa.rag.model_providers.embeddings.hash_embedding_provider import HashEmbeddingProvider


class DirectAgent:
    model = None

    def decide(self, state, _definitions):
        return AgentDecision("finish", final_answer=f"回答：{state['query']}")


def test_checkpointer_is_the_only_conversation_replay_source():
    workflow = DeclarationQAWorkflow(agent=DirectAgent())
    workflow.invoke(ChatRequest(query="第一轮", user_id=7, session_id="memory-thread"))

    restored = workflow._initial_state(
        ChatRequest(query="第二轮", user_id=7, session_id="memory-thread")
    )

    assert [(item["role"], item["content"]) for item in restored["agent_messages"]] == [
        ("user", "第一轮"),
        ("assistant", "回答：第一轮"),
    ]
    assert "short_memory" not in restored
    workflow.close()


def test_checkpointer_threads_are_isolated_by_user():
    workflow = DeclarationQAWorkflow(agent=DirectAgent())
    workflow.invoke(ChatRequest(query="用户一的内容", user_id=1, session_id="same-session"))

    other_user = workflow._initial_state(
        ChatRequest(query="用户二的问题", user_id=2, session_id="same-session")
    )

    assert other_user["agent_messages"] == []
    workflow.close()


def test_tool_observation_refreshes_input_without_resummarizing():
    workflow = DeclarationQAWorkflow(agent=DirectAgent())

    edges = {(edge.source, edge.target) for edge in workflow.graph.get_graph().edges}

    assert ("agent_observe", "agent_summarize") not in edges
    assert ("agent_observe", "agent_refresh_input") in edges
    assert ("agent_refresh_input", "agent_decide") in edges
    workflow.close()


def test_session_deletion_removes_matching_checkpoint_thread():
    workflow = DeclarationQAWorkflow(agent=DirectAgent())
    workflow.invoke(ChatRequest(query="需要删除的会话", user_id=9, session_id="delete-me"))

    assert workflow.delete_session(user_id=9, session_id="delete-me") is True
    restored = workflow._initial_state(
        ChatRequest(query="新问题", user_id=9, session_id="delete-me")
    )

    assert restored["agent_messages"] == []
    workflow.close()


def test_langmem_message_conversion_preserves_tool_call_pairs_and_ids():
    converted = message_dicts_to_langchain(
        [
            {"id": "u1", "role": "user", "content": "查询申请"},
            {
                "id": "a1",
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "ticket_list_query",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            },
            {
                "id": "t1",
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "ticket_list_query",
                "content": {"ok": True},
            },
        ]
    )

    assert isinstance(converted[0], HumanMessage)
    assert isinstance(converted[1], AIMessage)
    assert isinstance(converted[2], ToolMessage)
    assert [message.id for message in converted] == ["u1", "a1", "t1"]
    assert converted[1].tool_calls[0]["id"] == converted[2].tool_call_id == "call-1"


def test_sqlalchemy_postgres_url_is_normalized_for_psycopg():
    assert (
        psycopg_connection_url("postgresql+psycopg://user:pass@localhost:5432/db")
        == "postgresql://user:pass@localhost:5432/db"
    )


def test_langmem_creates_running_summary_after_token_threshold():
    runtime = build_test_memory_runtime(summary_max_tokens=200, summary_trigger_tokens=50)

    summarized, context = runtime.summarize(
        [
            {"id": "u1", "role": "user", "content": "较长的用户消息。" * 20},
            {"id": "a1", "role": "assistant", "content": "较长的回答。" * 20},
            {"id": "u2", "role": "user", "content": "第二轮问题"},
            {"id": "a2", "role": "assistant", "content": "第二轮回答"},
            {"id": "u3", "role": "user", "content": "第三轮问题"},
            {"id": "a3", "role": "assistant", "content": "第三轮回答"},
            {"id": "u4", "role": "user", "content": "第四轮问题"},
        ],
        {},
    )

    running_summary = context["running_summary"]
    assert running_summary.summary == "会话摘要"
    assert running_summary.summarized_message_ids == {"u1", "a1"}
    assert summarized[0].type == "system"
    assert [message.id for message in summarized[1:]] == [
        "u2",
        "a2",
        "u3",
        "a3",
        "u4",
    ]
    runtime.close()


def test_prepare_input_does_not_trigger_summary_after_tool_execution():
    runtime = build_test_memory_runtime(summary_max_tokens=200, summary_trigger_tokens=10)

    prepared, context = runtime.prepare_input(
        [
            {"id": "u1", "role": "user", "content": "较长的用户消息。" * 20},
            {"id": "a1", "role": "assistant", "content": "较长的回答。" * 20},
        ],
        {},
    )

    assert context == {}
    assert [message.id for message in prepared] == ["u1", "a1"]
    runtime.close()


def test_long_term_memory_is_isolated_by_user_namespace():
    runtime = build_test_memory_runtime()
    runtime.store.put(
        ("users", "1", "memories"),
        "preference-1",
        {"kind": "Memory", "content": {"content": "用户明确偏好简洁回答"}},
    )

    owner = runtime.retrieve(user_id=1, query="回答风格")
    other_user = runtime.retrieve(user_id=2, query="回答风格")

    assert owner["items"][0]["content"] == "用户明确偏好简洁回答"
    assert other_user["items"] == []
    runtime.close()


def build_test_memory_runtime(
    *,
    summary_max_tokens: int = 100,
    summary_trigger_tokens: int = 80,
) -> LangGraphMemoryRuntime:
    return LangGraphMemoryRuntime(
        resources=build_memory_resources(
            database_url=None,
            embedding_provider=HashEmbeddingProvider(),
        ),
        model=FakeListChatModel(responses=["会话摘要"]),
        summary_max_tokens=summary_max_tokens,
        summary_trigger_tokens=summary_trigger_tokens,
        summary_output_tokens=20,
        long_term_top_k=3,
        background_workers=1,
    )

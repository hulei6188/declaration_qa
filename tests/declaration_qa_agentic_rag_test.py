from __future__ import annotations

from declaration_qa.api.schemas import ChatRequest
from declaration_qa.application.tools import (
    KnowledgeSearchTool,
    TicketDetailQueryTool,
    TicketListQueryTool,
    ToolExecutionContext,
    ToolRegistry,
)
from declaration_qa.application.agent import AgentDecision, ToolCallingAgent, choose_application_id, rule_agent_decision
from declaration_qa.application.workflow import DeclarationQAWorkflow
from declaration_qa.application.nodes.agent_runtime import agent_tool_call_allowed, agent_tool_call_denial_reason
from declaration_qa.core.business import InMemoryBusinessDataAdapter
from declaration_qa.core.domain import KnowledgeChunk, RetrievalResult, RetrievedSource


class CapturingKnowledgeRag:
    rerank_score_threshold = 0.7
    rerank_max_sources = 36

    def __init__(self):
        self.calls = []
        chunk = KnowledgeChunk(
            id="chunk-materials",
            doc_id="doc-materials",
            parent_id="",
            chunk_type="child",
            title="健康证办理指南",
            content="申请健康证需要身份证明和体检材料。",
            region_code="340000",
            policy_version="2025",
        )
        self.result = RetrievalResult(
            sources=[RetrievedSource(chunk=chunk, score=0.9, rerank_score=0.91, retrieval_channel="hybrid")],
            filter={},
            status={"retrieval_mode": "test", "score_cutoff": {"threshold": 0.7}},
        )

    def retrieve_rewrites(self, original_query, retrieval_queries, filters, *, top_k, retrieval_plan):
        self.calls.append(
            {
                "query": original_query,
                "queries": retrieval_queries,
                "filters": dict(filters),
                "retrieval_plan": dict(retrieval_plan),
            }
        )
        return RetrievalResult(self.result.sources, dict(filters), self.result.status)

    def retrieve(self, query, filters, *, top_k, retrieval_plan):
        self.calls.append(
            {"query": query, "queries": [], "filters": dict(filters), "retrieval_plan": dict(retrieval_plan)}
        )
        return RetrievalResult(self.result.sources, dict(filters), self.result.status)

    def get_chunks_by_ids(self, chunk_ids):
        return [source.chunk for source in self.result.sources if source.chunk.id in chunk_ids]


class HistoricalEvidenceRag(CapturingKnowledgeRag):
    def __init__(self):
        super().__init__()
        self.history_chunk = KnowledgeChunk(
            id="chunk-history",
            doc_id="doc-history",
            parent_id="",
            chunk_type="child",
            title="取水许可办事指南",
            content="审查阶段包括材料核验和现场评估。",
            region_code="340000",
            policy_version="2025",
            status="published",
        )
        self.rerank_inputs = []

    def get_chunks_by_ids(self, chunk_ids):
        return [self.history_chunk] if self.history_chunk.id in chunk_ids else []

    def rerank_candidates(self, query, sources, filters, *, retrieval_plan=None):
        self.rerank_inputs.append([source.chunk.id for source in sources])
        return sorted(sources, key=lambda source: source.chunk.id)


class ScriptedAgent:
    """用显式 Tool Calling 脚本隔离工作流测试，不依赖关键词意图规则。"""

    def __init__(self, decide):
        self._decide = decide

    def decide(self, state, tool_definitions):
        return self._decide(state, tool_definitions)


def tool_call(call_id, name, arguments=None):
    return {"id": call_id, "name": name, "arguments": arguments or {}}


def knowledge_arguments(text):
    return {"queries": [{"text": text, "strategy": "hybrid"}]}


def test_model_error_returns_service_failure_instead_of_raw_evidence():
    raw_evidence = "| 字段 | 内容 |\n| --- | --- |\n| 材料依据 | 原始证据正文 |"

    decision = rule_agent_decision(
        {"retrieved_sources": [{"title": "办事指南", "evidence_text": raw_evidence}]},
        model_error=TimeoutError("timed out"),
    )

    assert decision.reason == "agent_model_error:TimeoutError"
    assert decision.final_answer == "已检索到相关依据，但回答生成服务暂时不可用，请稍后重试。"
    assert raw_evidence not in decision.final_answer


def test_tool_registry_exposes_item_scoped_ticket_tools_without_user_id_argument():
    adapter = InMemoryBusinessDataAdapter()
    registry = ToolRegistry()
    registry.register(TicketListQueryTool(adapter))
    registry.register(TicketDetailQueryTool(adapter))

    definitions = {item["function"]["name"]: item["function"] for item in registry.definitions()}

    assert set(definitions) == {"ticket_list_query", "ticket_detail_query"}
    assert "user_id" not in definitions["ticket_list_query"]["parameters"].get("properties", {})
    assert "region_code" not in definitions["ticket_list_query"]["parameters"].get("properties", {})
    assert "user_id" not in definitions["ticket_detail_query"]["parameters"].get("properties", {})
    assert "ticket_id" not in definitions["ticket_detail_query"]["parameters"].get("properties", {})

    context = ToolExecutionContext(user_id=1, region_code="340000")
    listing = registry.execute("ticket_list_query", {}, context, call_id="list-1")
    detail = registry.execute(
        "ticket_detail_query",
        {"item_code": "ITEM-RESIDENCE-PERMIT", "detail_type": "rejection"},
        context,
        call_id="detail-1",
    )

    assert listing.ok is True
    assert any(item["item_code"] == "ITEM-RESIDENCE-PERMIT" for item in listing.data["items"])
    assert detail.ok is True
    assert detail.data["applications"][0]["application_id"] == 2002
    assert len(detail.data["applications"][0]["status_history"]) == 2
    assert detail.data["applications"][0]["matching_status_events"][0]["reason"] == "缺少有效居住证明材料"


def test_knowledge_tool_schema_requires_structured_agent_queries():
    registry = ToolRegistry()
    registry.register(KnowledgeSearchTool(lambda request, context: {"sources": [], "executed_queries": []}))

    parameters = registry.definitions()[0]["function"]["parameters"]

    assert set(parameters["properties"]) == {"queries"}
    assert "queries" in parameters["properties"]
    assert "query_focus" not in parameters["properties"]
    assert "evidence_refs" not in parameters["properties"]
    assert "chunk_ids" not in parameters["properties"]
    assert "user_id" not in parameters["properties"]
    assert "top_k" not in parameters["properties"]


def test_single_agent_can_search_knowledge_after_observing_ticket_results():
    rag = CapturingKnowledgeRag()

    def decide(state, _definitions):
        completed = {item["tool_name"] for item in state.get("tool_results") or [] if item.get("ok")}
        if "ticket_list_query" not in completed:
            return AgentDecision("tools", [tool_call("list", "ticket_list_query")])
        if "knowledge_search" not in completed:
            return AgentDecision(
                "tools",
                [tool_call("knowledge", "knowledge_search", knowledge_arguments("居住证补正材料要求"))],
            )
        return AgentDecision("finish", final_answer="已结合申请事实和公开政策回答。[业务1][证据1]")

    response = DeclarationQAWorkflow(rag_engine=rag, agent=ScriptedAgent(decide)).invoke(
        ChatRequest(query="我的居住证申请怎么补正", user_id=1)
    )

    assert [item["tool_name"] for item in response.tool_results] == [
        "ticket_list_query",
        "knowledge_search",
    ]
    assert response.answer_mode == "tool_grounded"
    assert response.answer == "已结合申请事实和公开政策回答。[业务1][证据1]"


def test_single_agent_replays_assistant_and_tool_messages_by_call_id():
    captured_states = []

    def decide(state, _definitions):
        captured_states.append(list(state.get("agent_messages") or []))
        if not state.get("tool_results"):
            return AgentDecision("tools", [tool_call("list-call", "ticket_list_query")])
        return AgentDecision("finish", final_answer="已查询申请列表。[业务1]")

    response = DeclarationQAWorkflow(
        rag_engine=CapturingKnowledgeRag(),
        agent=ScriptedAgent(decide),
    ).invoke(ChatRequest(query="我有哪些申请", user_id=1))

    replay = captured_states[-1]
    assert replay[0]["role"] == "user"
    assistant = next(item for item in replay if item.get("role") == "assistant")
    tool = next(item for item in replay if item.get("role") == "tool")
    assert assistant["tool_calls"][0]["id"] == "list-call"
    assert tool["tool_call_id"] == "list-call"
    assert response.answer == "已查询申请列表。[业务1]"


def test_agent_prepare_preserves_checkpoint_summary_context():
    workflow = DeclarationQAWorkflow(
        rag_engine=CapturingKnowledgeRag(),
        agent=ScriptedAgent(lambda state, definitions: AgentDecision("finish", final_answer="直接回答")),
    )
    state = workflow._initial_state(ChatRequest(query="继续", user_id=1))
    state["normalized_query"] = "继续"
    state["memory_context"] = {"summary_marker": "kept"}

    prepared = workflow.agent_prepare(state)

    assert prepared["memory_context"]["summary_marker"] == "kept"


def test_chat_request_has_no_bound_ticket_fields():
    fields = ChatRequest.model_json_schema().get("properties", {})

    assert "apply_id" not in fields
    assert "apply_context" not in fields


def test_ticket_detail_returns_every_application_and_every_status_change_for_item():
    adapter = InMemoryBusinessDataAdapter()
    previous = dict(adapter._applications["A2002"])
    previous.update(
        {
            "application_id": 1900,
            "status": "completed",
            "current_node": "FINISH",
            "node_name": "办结",
            "submitted_at": "2025-03-02T08:40:00+08:00",
            "completed_at": "2025-03-20T16:10:00+08:00",
            "updated_at": "2025-03-20T16:10:00+08:00",
            "status_history": [
                {"event_id": 19001, "sequence_no": 1, "from_status": None, "status": "submitted", "changed_at": "2025-03-02T08:40:00+08:00"},
                {"event_id": 19002, "sequence_no": 2, "from_status": "submitted", "status": "accepted", "changed_at": "2025-03-03T09:20:00+08:00"},
                {"event_id": 19003, "sequence_no": 3, "from_status": "accepted", "status": "completed", "changed_at": "2025-03-20T16:10:00+08:00"},
            ],
        }
    )
    adapter._applications["A1900"] = previous

    applications = adapter.get_applications_by_item(user_id=1, item_code="ITEM-RESIDENCE-PERMIT")

    assert {item["application_id"] for item in applications} == {1900, 2002}
    assert [event["sequence_no"] for event in next(item for item in applications if item["application_id"] == 1900)["status_history"]] == [1, 2, 3]
    assert choose_application_id("看一下2025年那次申请", applications) == 1900
    assert choose_application_id("看一下上一次申请", applications) == 1900


def test_ticket_detail_is_scoped_to_trusted_user_context():
    registry = ToolRegistry()
    registry.register(TicketDetailQueryTool(InMemoryBusinessDataAdapter()))

    result = registry.execute(
        "ticket_detail_query",
        {"item_code": "ITEM-RESIDENCE-PERMIT"},
        ToolExecutionContext(user_id=999, region_code="340000"),
    )

    assert result.ok is False
    assert result.status == "not_found"
    assert result.data["applications"] == []


def test_tool_policy_allows_direct_business_and_knowledge_queries():
    assert agent_tool_call_allowed("ticket_list_query", "健康证需要哪些材料") is True
    assert agent_tool_call_allowed("knowledge_search", "我的居住证申请进度") is True
    assert agent_tool_call_allowed("knowledge_search", "我的居住证申请被退回，怎么补正") is True
    assert agent_tool_call_denial_reason(
        "ticket_detail_query",
        "我的居住证申请进度",
        arguments={"item_code": "ITEM-RESIDENCE-PERMIT"},
    ) == ""
    assert agent_tool_call_denial_reason(
        "ticket_detail_query",
        "我的居住证申请进度",
        arguments={"item_code": "ITEM-OTHER"},
    ) == ""
    assert agent_tool_call_denial_reason(
        "knowledge_search",
        "我的居住证申请被退回，怎么补正",
        arguments={"queries": [{"text": "居住证补正材料", "strategy": "hybrid"}]},
    ) == ""


def test_agent_ticket_question_selects_item_and_application_without_bound_ticket():
    def decide(state, _definitions):
        completed = {item["tool_name"] for item in state.get("tool_results") or [] if item.get("ok")}
        if "ticket_list_query" not in completed:
            return AgentDecision("tools", [tool_call("list", "ticket_list_query")])
        if "ticket_detail_query" not in completed:
            return AgentDecision(
                "tools",
                [
                    tool_call(
                        "detail",
                        "ticket_detail_query",
                        {"item_code": "ITEM-RESIDENCE-PERMIT", "detail_type": "rejection"},
                    )
                ],
            )
        return AgentDecision("finish", final_answer="该申请因缺少有效居住证明材料被退回。[业务2]")

    workflow = DeclarationQAWorkflow(
        rag_engine=CapturingKnowledgeRag(),
        agent=ScriptedAgent(decide),
    )

    response = workflow.invoke(
        ChatRequest(
            query="我的居住证申请为什么被退回",
            user_id=1,
            region_code="340000",
        )
    )

    tool_names = [item["tool_name"] for item in response.tool_results]
    assert tool_names == ["ticket_list_query", "ticket_detail_query"]
    assert response.route == "agentic_rag"
    assert response.intent == "agentic_rag"
    assert response.business_snapshot["item_code"] == "ITEM-RESIDENCE-PERMIT"
    assert response.business_snapshot["selected_application_id"] == 2002
    assert "缺少有效居住证明材料" in response.answer


def test_agent_can_answer_application_list_without_forcing_a_detail_selection():
    def decide(state, _definitions):
        if not state.get("tool_results"):
            return AgentDecision("tools", [tool_call("list", "ticket_list_query")])
        return AgentDecision(
            "finish",
            final_answer="你有健康证办理和居住证办理申请记录。[业务1]",
        )

    workflow = DeclarationQAWorkflow(
        rag_engine=CapturingKnowledgeRag(),
        agent=ScriptedAgent(decide),
    )

    response = workflow.invoke(ChatRequest(query="我有哪些历史申请", user_id=1, region_code="340000"))

    assert response.intent == "agentic_rag"
    assert [item["tool_name"] for item in response.tool_results] == ["ticket_list_query"]
    assert response.business_snapshot == {}
    assert "健康证办理" in response.answer
    assert "居住证办理" in response.answer


def test_agent_knowledge_question_uses_only_knowledge_tool_and_frozen_query():
    rag = CapturingKnowledgeRag()

    def decide(state, _definitions):
        if not state.get("tool_results"):
            return AgentDecision(
                "tools",
                [tool_call("knowledge", "knowledge_search", knowledge_arguments("健康证需要哪些材料"))],
            )
        return AgentDecision("finish", final_answer="申请健康证需要身份证明和体检材料。[证据1]")

    workflow = DeclarationQAWorkflow(rag_engine=rag, agent=ScriptedAgent(decide))

    response = workflow.invoke(
        ChatRequest(
            query="健康证需要哪些材料",
            user_id=1,
            region_code="340000",
            metadata_filter={"region_code": "340000"},
        )
    )

    assert [item["tool_name"] for item in response.tool_results] == ["knowledge_search"]
    assert response.sources[0].chunk_id == "chunk-materials"
    assert rag.calls
    assert rag.calls[0]["query"] == "健康证需要哪些材料"
    assert rag.calls[0]["filters"] == {"region_code": "340000"}
    assert response.agent_query_plan["executed_queries"][0]["strategy"] == "hybrid"
    assert "ticket_fact_type" not in rag.calls[0]["retrieval_plan"]
    assert "A2002" not in str(rag.calls[0])


def test_mixed_ticket_and_guidance_question_keeps_business_result_out_of_rag_query():
    rag = CapturingKnowledgeRag()

    def decide(state, _definitions):
        completed = {item["tool_name"] for item in state.get("tool_results") or [] if item.get("ok")}
        if not completed:
            return AgentDecision(
                "tools",
                [
                    tool_call("list", "ticket_list_query"),
                    tool_call(
                        "knowledge",
                        "knowledge_search",
                        knowledge_arguments("我的居住证申请为什么被退回，应该怎么补正"),
                    ),
                ],
            )
        if "ticket_detail_query" not in completed:
            return AgentDecision(
                "tools",
                [
                    tool_call(
                        "detail",
                        "ticket_detail_query",
                        {"item_code": "ITEM-RESIDENCE-PERMIT", "detail_type": "rejection"},
                    )
                ],
            )
        return AgentDecision("finish", final_answer="请根据退回原因补交有效居住证明材料。[业务2][证据1]")

    workflow = DeclarationQAWorkflow(rag_engine=rag, agent=ScriptedAgent(decide))

    response = workflow.invoke(
        ChatRequest(
            query="我的居住证申请为什么被退回，应该怎么补正",
            user_id=1,
            region_code="340000",
        )
    )

    assert [item["tool_name"] for item in response.tool_results] == [
        "ticket_list_query",
        "knowledge_search",
        "ticket_detail_query",
    ]
    assert rag.calls[0]["query"] == "我的居住证申请为什么被退回，应该怎么补正"
    serialized_call = str(rag.calls[0])
    assert "缺少有效居住证明材料" not in serialized_call
    assert "MISSING_RESIDENCE_PROOF" not in serialized_call


def test_agent_final_response_is_used_without_separate_answer_generation():
    class FinalAnswerAgent:
        def decide(self, state, tool_definitions):
            if not state.get("tool_results"):
                return AgentDecision(
                    "tools",
                    [
                        {
                            "id": "knowledge",
                            "name": "knowledge_search",
                            "arguments": {
                                "queries": [
                                        {"text": "健康证申请材料", "strategy": "hybrid"}
                                ]
                            },
                        }
                    ],
                )
            return AgentDecision(
                "finish",
                provider="fake_agent",
                final_answer="这是 Agent 根据知识工具证据生成的最终答案。[证据1]",
                final_prompt="agent prompt",
            )

    workflow = DeclarationQAWorkflow(rag_engine=CapturingKnowledgeRag(), agent=FinalAnswerAgent())

    response = workflow.invoke(ChatRequest(query="健康证需要哪些材料", user_id=1))
    audit = workflow.audit_store.get(response.audit_id)

    assert response.answer == "这是 Agent 根据知识工具证据生成的最终答案。[证据1]"
    assert response.evidence_decision["ok"] is True
    assert audit["evidence_ok"] is True
    assert audit["evidence_decision"]["ok"] is True
    assert response.agent_query_plan["requested_queries"][0] == {
        "text": "健康证申请材料",
        "strategy": "hybrid",
    }


def test_agent_without_tool_call_can_finish_directly_without_finalizer_override():
    class DirectAgent:
        def __init__(self):
            self.finalize_calls = 0

        def decide(self, state, tool_definitions):
            return AgentDecision(
                "finish",
                provider="native_tool_calling",
                final_answer="1+1等于2。",
                final_prompt="direct prompt",
            )

        def finalize(self, state):
            self.finalize_calls += 1
            raise AssertionError("direct answer must not be regenerated")

    agent = DirectAgent()
    workflow = DeclarationQAWorkflow(rag_engine=CapturingKnowledgeRag(), agent=agent)

    response = workflow.invoke(ChatRequest(query="1+1等于几", user_id=1))

    assert response.answer == "1+1等于2。"
    assert response.tool_results == []
    assert response.refused is False
    assert response.answer_mode == "direct"
    assert agent.finalize_calls == 0


def test_empty_agent_answer_does_not_bypass_evidence_gate():
    workflow = DeclarationQAWorkflow(
        rag_engine=CapturingKnowledgeRag(),
        agent=ScriptedAgent(lambda state, definitions: AgentDecision("finish")),
        agent_max_iterations=1,
    )

    response = workflow.invoke(ChatRequest(query="你好", user_id=1))

    assert response.refused is True
    assert response.answer
    assert response.evidence_decision.get("mode") != "direct"


def test_agent_answer_after_tool_results_must_pass_the_evidence_gate():
    workflow = DeclarationQAWorkflow(
        rag_engine=CapturingKnowledgeRag(),
        agent=ScriptedAgent(
            lambda state, definitions: AgentDecision(
                "finish",
                final_answer="没有证据也直接回答。",
            )
        ),
    )
    state = workflow._initial_state(ChatRequest(query="健康证办理材料", user_id=1))
    state = workflow.query_preprocess(state)
    state = workflow.agent_prepare(state)
    state["tool_results"] = [{"tool_name": "knowledge_search", "ok": True, "data": {}}]

    decided = workflow.agent_decide(state)

    assert decided["answer_mode"] == "tool_grounded"
    assert workflow._agent_decision_route(decided) == "verify"
    verified = workflow.verify_agent_answer(decided)
    assert verified["evidence_ok"] is False
    assert verified["evidence_decision"]["reason"] == "no_traceable_knowledge_evidence"


def test_single_agent_finishes_a_compound_question_without_worker_decomposition():
    class CompoundDirectAgent:
        def decide(self, state, tool_definitions):
            return AgentDecision(
                "finish",
                provider="native_tool_calling",
                final_answer="1+1等于2，2+2等于4。",
            )

    workflow = DeclarationQAWorkflow(rag_engine=CapturingKnowledgeRag(), agent=CompoundDirectAgent())

    response = workflow.invoke(ChatRequest(query="1+1等于几，同时2+2等于几", user_id=1))

    assert response.refused is False
    assert response.tool_results == []
    assert "1+1等于2" in response.answer
    assert "2+2等于4" in response.answer
    assert response.answer_mode == "direct"
    assert not any(event["event"].startswith("knowledge_worker") for event in response.events)


def test_single_agent_never_forces_tool_choice_required():
    class Response:
        content = "直接回答"
        tool_calls = []
        response_metadata = {}
        usage_metadata = {}

    class BoundModel:
        def invoke(self, messages, config=None):
            return Response()

    class CapturingModel:
        model_name = "test-model"

        def __init__(self):
            self.bind_kwargs = []

        def bind_tools(self, definitions, **kwargs):
            self.bind_kwargs.append(kwargs)
            return BoundModel()

    model = CapturingModel()
    agent = ToolCallingAgent(model)
    definitions = [{"type": "function", "function": {"name": "knowledge_search"}}]

    first = agent.decide({"query": "你好"}, definitions)
    retry = agent.decide({"query": "办理材料", "tool_results": []}, definitions)

    assert first.final_answer == "直接回答"
    assert model.bind_kwargs[0] == {}
    assert model.bind_kwargs[1] == {}
    assert retry.final_answer == "直接回答"


def test_knowledge_search_uses_only_current_search_sources():
    rag = HistoricalEvidenceRag()

    def decide(state, definitions):
        if not state.get("tool_results"):
            return AgentDecision(
                "tools",
                [tool_call("knowledge", "knowledge_search", knowledge_arguments("华侨回国定居审批"))],
            )
        return AgentDecision("finish", final_answer="已根据本轮证据回答。[证据1]")

    workflow = DeclarationQAWorkflow(rag_engine=rag, agent=ScriptedAgent(decide))
    initial = workflow._initial_state(
        ChatRequest(query="华侨回国定居审批呢", user_id=1, session_id="session-history")
    )
    result = workflow.graph.invoke(
        initial,
        config=workflow._graph_config(user_id=1, session_id="session-history"),
    )

    assert {source["chunk_id"] for source in result["retrieved_sources"]} == {"chunk-materials"}
    assert result["rag_status"]["historical_merge_mode"] == "current_search_only"


def test_follow_up_uses_previous_full_chunk_without_new_vector_search():
    rag = CapturingKnowledgeRag()

    def decide(state, definitions):
        if not state.get("retrieved_sources"):
            return AgentDecision(
                "tools",
                [tool_call("knowledge", "knowledge_search", knowledge_arguments("健康证申请材料"))],
            )
        return AgentDecision("finish", final_answer="第二项是体检材料。[证据1]")

    workflow = DeclarationQAWorkflow(rag_engine=rag, agent=ScriptedAgent(decide))
    first = workflow.invoke(
        ChatRequest(query="健康证需要哪些材料", user_id=1, session_id="memory-knowledge")
    )
    first_call_count = len(rag.calls)
    second = workflow.invoke(
        ChatRequest(query="第二项呢", user_id=1, session_id="memory-knowledge")
    )

    assert first.refused is False
    assert second.refused is False
    assert len(rag.calls) == first_call_count
    assert second.evidence_decision["evidence_registry"]["knowledge"][0]["origin"] == "recent_tool_context"
    restored = workflow._initial_state(
        ChatRequest(query="第三轮", user_id=1, session_id="memory-knowledge")
    )
    tool_messages = [item for item in restored["agent_messages"] if item.get("role") == "tool"]
    assert tool_messages[0]["content"]["data"]["sources"][0]["evidence_text"] == "申请健康证需要身份证明和体检材料。"


def test_full_knowledge_evidence_remains_available_before_rolling_summary():
    rag = CapturingKnowledgeRag()

    def decide(state, definitions):
        if not state.get("retrieved_sources"):
            return AgentDecision(
                "tools",
                [tool_call("knowledge", "knowledge_search", knowledge_arguments("健康证申请材料"))],
            )
        return AgentDecision("finish", final_answer="根据窗口内完整证据回答。[证据1]")

    workflow = DeclarationQAWorkflow(rag_engine=rag, agent=ScriptedAgent(decide))
    session_id = "memory-three-turn-window"

    workflow.invoke(ChatRequest(query="第一轮检索", user_id=1, session_id=session_id))
    workflow.invoke(ChatRequest(query="第二轮追问", user_id=1, session_id=session_id))
    workflow.invoke(ChatRequest(query="第三轮追问", user_id=1, session_id=session_id))
    assert len(rag.calls) == 1

    fourth = workflow.invoke(ChatRequest(query="第四轮追问", user_id=1, session_id=session_id))

    assert len(rag.calls) == 1
    assert fourth.evidence_decision["evidence_registry"]["knowledge"][0]["origin"] == "recent_tool_context"
    assert not any(
        item.get("role") == "tool"
        and item.get("content", {}).get("memory_trust") == "reference_only"
        for item in workflow._initial_state(
            ChatRequest(query="检查", user_id=1, session_id=session_id)
        )["agent_messages"]
    )
    assert fourth.refused is False


def test_policy_version_change_runs_new_search_instead_of_reference_recall():
    rag = CapturingKnowledgeRag()

    def decide(state, definitions):
        has_current_search = any(
            result.get("tool_name") == "knowledge_search"
            for result in state.get("tool_results") or []
            if isinstance(result, dict)
        )
        if not has_current_search:
            return AgentDecision(
                "tools",
                [tool_call("knowledge", "knowledge_search", knowledge_arguments("健康证申请材料"))],
            )
        return AgentDecision("finish", final_answer="已根据本轮检索结果回答。[证据1]")

    workflow = DeclarationQAWorkflow(rag_engine=rag, agent=ScriptedAgent(decide))
    workflow.invoke(
        ChatRequest(
            query="健康证需要哪些材料",
            user_id=1,
            session_id="memory-policy-version",
            policy_version="2025",
        )
    )
    first_call_count = len(rag.calls)
    second = workflow.invoke(
        ChatRequest(
            query="第二项呢",
            user_id=1,
            session_id="memory-policy-version",
            policy_version="2026",
        )
    )

    assert len(rag.calls) == first_call_count + 1
    assert not any(event["event"] == "memory_recalled" for event in second.events)


def test_topic_shift_can_search_again_without_reference_recall():
    rag = CapturingKnowledgeRag()

    def decide(state, definitions):
        has_current_search = any(
            result.get("tool_name") == "knowledge_search"
            for result in state.get("tool_results") or []
            if isinstance(result, dict)
        )
        if not has_current_search:
            query = str(state.get("query") or "")
            return AgentDecision(
                "tools",
                [tool_call("knowledge", "knowledge_search", knowledge_arguments(query))],
            )
        return AgentDecision("finish", final_answer="已根据本轮证据回答。[证据1]")

    workflow = DeclarationQAWorkflow(rag_engine=rag, agent=ScriptedAgent(decide))
    workflow.invoke(
        ChatRequest(query="健康证需要哪些材料", user_id=1, session_id="memory-topic-shift")
    )
    second = workflow.invoke(
        ChatRequest(query="居住证怎么办理", user_id=1, session_id="memory-topic-shift")
    )

    assert len(rag.calls) == 2
    assert not any(event["event"] == "memory_recalled" for event in second.events)


def test_business_memory_only_replays_for_explicit_historical_question():
    calls = []

    def decide(state, definitions):
        if not state.get("tool_results"):
            call_id = f"queue-{len(calls) + 1}"
            calls.append(call_id)
            return AgentDecision(
                "tools",
                [tool_call(call_id, "queue_status_query", {"appointment_id": 4101})],
            )
        result = state["tool_results"][0]
        if result.get("memory_origin") == "historical_snapshot":
            return AgentDecision("finish", final_answer="刚才查询时间的排队结果按当时快照展示。[业务1]")
        return AgentDecision("finish", final_answer="当前排队结果已查询。[业务1]")

    workflow = DeclarationQAWorkflow(
        rag_engine=CapturingKnowledgeRag(),
        business_adapter=InMemoryBusinessDataAdapter(),
        agent=ScriptedAgent(decide),
    )
    first = workflow.invoke(
        ChatRequest(query="我的排队情况", user_id=1, session_id="memory-business")
    )
    historical = workflow.invoke(
        ChatRequest(query="刚才显示的排队情况是什么", user_id=1, session_id="memory-business")
    )
    current = workflow.invoke(
        ChatRequest(query="现在排队情况是什么", user_id=1, session_id="memory-business")
    )

    assert first.refused is False
    assert historical.refused is False
    assert historical.tool_results[0]["memory_origin"] == "historical_snapshot"
    assert historical.evidence_decision["evidence_registry"]["business"][0]["origin"] == "historical_snapshot"
    assert len(calls) == 2
    assert current.tool_results[0]["call_id"] == "queue-2"

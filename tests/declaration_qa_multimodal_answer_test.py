from types import SimpleNamespace

from declaration_qa.application.agent import ToolCallingAgent
from declaration_qa.rag.model_providers.answers import OpenAIAnswerGenerator


class CapturingModel:
    def __init__(self, response="模型回答"):
        self.response = response
        self.inputs = []

    def invoke(self, value, config=None):
        self.inputs.append(value)
        return SimpleNamespace(content=self.response)


def make_generator(tmp_path):
    generator = OpenAIAnswerGenerator.__new__(OpenAIAnswerGenerator)
    generator._llm = CapturingModel("文本回答")
    generator._vl_llm = CapturingModel("多模态回答")
    generator.model = "qwen-plus"
    generator.vl_model = "qwen3-vl-plus"
    generator.vl_max_evidence_images = 3
    generator.vl_min_image_score = 0.7
    generator.vl_max_image_bytes = 1024 * 1024
    generator.allowed_image_dirs = [tmp_path.resolve()]
    return generator


def test_user_image_and_deduplicated_evidence_images_are_sent_to_vl(tmp_path):
    query_image = tmp_path / "query.png"
    query_image.write_bytes(b"query-image")
    evidence_image = tmp_path / "evidence.png"
    evidence_image.write_bytes(b"evidence-image")
    generator = make_generator(tmp_path)
    image_source = {
        "chunk_id": "image-1",
        "title": "取水许可流程图",
        "evidence_text": "窗口受理后转交水资源管理处承办。",
        "score": 0.82,
        "rerank_score": 0.91,
        "metadata": {
            "modality": "image",
            "asset_id": "asset-flow",
            "asset_path": str(evidence_image),
            "asset_content_type": "image/png",
            "attachment_page_no": 1,
            "visual_summary": "取水许可流程图",
        },
    }
    duplicate_source = {
        **image_source,
        "chunk_id": "image-duplicate",
        "rerank_score": 0.88,
    }
    state = {
        "query": "这张图对应什么办理流程？",
        "intent": "process_steps",
        "retrieval_chain": "image_dense",
        "query_image": {
            "file_path": str(query_image),
            "filename": "用户流程图.png",
            "content_type": "image/png",
        },
        "retrieved_sources": [image_source, duplicate_source],
    }

    plan = generator.generation_plan(state)
    answer, prompt = generator.generate(state)

    assert plan == {
        "mode": "vl",
        "model": "qwen3-vl-plus",
        "has_query_image": True,
        "query_image_filename": "用户流程图.png",
        "evidence_image_count": 1,
        "evidence_chunk_ids": ["image-1"],
    }
    assert answer == "多模态回答"
    assert "这张图对应什么办理流程" in prompt
    assert not generator._llm.inputs
    messages = generator._vl_llm.inputs[0]
    content = messages[0].content
    image_blocks = [item for item in content if item.get("type") == "image_url"]
    assert len(image_blocks) == 2
    assert all(item["image_url"]["url"].startswith("data:image/png;base64,") for item in image_blocks)
    text = "\n".join(item.get("text", "") for item in content if item.get("type") == "text")
    assert "用户查询图片" in text
    assert "知识库图片证据 1" in text
    assert "不能单独作为政策依据" in text


def test_text_only_answer_keeps_text_model(tmp_path):
    generator = make_generator(tmp_path)
    state = {
        "query": "取水许可需要哪些材料？",
        "intent": "material_requirements",
        "retrieved_sources": [
            {
                "chunk_id": "text-1",
                "title": "取水许可办事指南",
                "evidence_text": "申请材料包括取水许可申请书。",
                "score": 0.9,
                "metadata": {"modality": "text"},
            }
        ],
    }

    answer, _ = generator.generate(state)

    assert answer == "文本回答"
    assert generator._llm.inputs
    assert not generator._vl_llm.inputs


def test_single_agent_turn_receives_user_and_retrieved_images(tmp_path):
    query_image = tmp_path / "query.png"
    query_image.write_bytes(b"query-image")
    evidence_image = tmp_path / "evidence.png"
    evidence_image.write_bytes(b"evidence-image")

    class Response:
        content = "已结合图片回答。"
        tool_calls = []
        response_metadata = {}
        usage_metadata = {}

    class Model:
        model_name = "qwen3-vl-plus"

        def __init__(self):
            self.inputs = []

        def bind_tools(self, definitions):
            return self

        def invoke(self, messages, config=None):
            self.inputs.append(messages)
            return Response()

    model = Model()
    agent = ToolCallingAgent(
        model,
        multimodal_enabled=True,
        allowed_image_dirs=[str(tmp_path)],
        max_evidence_images=3,
        min_image_score=0.7,
    )
    source = {
        "chunk_id": "image-1",
        "title": "取水许可流程图",
        "evidence_text": "窗口受理后转交水资源管理处承办。",
        "rerank_score": 0.91,
        "metadata": {
            "modality": "image",
            "asset_id": "asset-flow",
            "asset_path": str(evidence_image),
            "asset_content_type": "image/png",
        },
    }
    state = {
        "query": "这张图对应什么办理流程？",
        "normalized_query": "这张图对应什么办理流程？",
        "query_image": {
            "file_path": str(query_image),
            "filename": "用户流程图.png",
            "content_type": "image/png",
        },
        "retrieved_sources": [source, {**source, "chunk_id": "image-duplicate"}],
        "agent_messages": [],
    }

    decision = agent.decide(state, [{"type": "function", "function": {"name": "knowledge_search"}}])

    assert decision.final_answer == "已结合图片回答。"
    content = model.inputs[0][-1].content
    image_blocks = [item for item in content if item.get("type") == "image_url"]
    assert len(image_blocks) == 2
    text = "\n".join(item.get("text", "") for item in content if item.get("type") == "text")
    assert "用户查询图片" in text
    assert "知识库图片证据 1" in text

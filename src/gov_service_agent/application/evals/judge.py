"""使用独立模型执行回答质量裁判。"""

from __future__ import annotations

import json
from typing import Any

from gov_service_agent.rag.router import parse_json_object


class IndependentJudge:
    """对答案忠实度、相关性、完整性和引用质量进行结构化评分。"""

    def __init__(self, model: Any | None, *, model_name: str = "") -> None:
        self.model = model
        self.model_name = model_name

    def evaluate(
        self,
        *,
        query: str,
        answer: str,
        contexts: list[str],
        reference_answer: str,
    ) -> dict[str, Any]:
        if self.model is None:
            return {"scores": {}, "status": {"status": "unavailable", "reason": "judge_not_configured"}, "records": []}
        prompt = f"""你是独立评测裁判，只根据给定证据评分，不补充外部知识。
请输出单个JSON对象，字段为 faithfulness、answer_relevancy、completeness、citation_precision、citation_coverage，
每项为0到1的小数；另有 reasons 对象解释每项，不得输出Markdown。

用户问题：{query}
候选回答：{answer}
参考答案：{reference_answer or '[无]'}
证据：{json.dumps(contexts, ensure_ascii=False)}
"""
        try:
            response = self.model.invoke(prompt)
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = "".join(str(item.get("text") or "") if isinstance(item, dict) else str(item) for item in content)
            parsed = parse_json_object(str(content or ""))
            reasons = dict(parsed.get("reasons") or {})
            scores = {}
            records = []
            for name in ("faithfulness", "answer_relevancy", "completeness", "citation_precision", "citation_coverage"):
                value = max(0.0, min(1.0, float(parsed.get(name))))
                scores[name] = value
                records.append(
                    {
                        "metric_name": name,
                        "evaluator_type": "llm_judge",
                        "evaluator_version": self.model_name or "independent-judge-v1",
                        "score": value,
                        "label": "pass" if value >= 0.95 else "fail",
                        "reason": str(reasons.get(name) or ""),
                        "raw_result": parsed,
                    }
                )
            return {"scores": scores, "status": {"status": "success", "model": self.model_name}, "records": records}
        except Exception as exc:
            return {
                "scores": {},
                "status": {"status": "failed", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"},
                "records": [],
            }

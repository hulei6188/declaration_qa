"""Ragas single-turn evaluation adapter isolated from the online request path."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RagasCaseResult:
    """Per-case Ragas scores and non-fatal metric errors."""

    scores: dict[str, float]
    status: dict[str, Any]


class RagasEvaluator:
    """Evaluate one answer with the model and embeddings already used by the service."""

    def __init__(
        self,
        *,
        model: Any,
        embedding_provider: Any,
        timeout_seconds: int = 60,
        max_retries: int = 1,
    ) -> None:
        if model is None:
            raise ValueError("Ragas evaluation requires an Agent LLM")
        self.model = model
        self.embedding_provider = embedding_provider
        self.timeout_seconds = max(1, int(timeout_seconds or 60))
        self.max_retries = max(0, int(max_retries or 0))

    def evaluate(
        self,
        *,
        query: str,
        answer: str,
        retrieved_contexts: list[str],
        reference_answer: str = "",
        reference_contexts: list[str] | None = None,
    ) -> RagasCaseResult:
        components = load_ragas_components()
        run_config = components["RunConfig"](
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
            max_workers=1,
        )
        llm = components["LangchainLLMWrapper"](self.model, run_config=run_config)
        embeddings = components["LangchainEmbeddingsWrapper"](
            LangChainEmbeddingAdapter(self.embedding_provider),
            run_config=run_config,
        )
        sample = components["SingleTurnSample"](
            user_input=str(query or ""),
            response=str(answer or ""),
            retrieved_contexts=[str(item) for item in retrieved_contexts if str(item).strip()],
            reference=str(reference_answer or "") or None,
            reference_contexts=[str(item) for item in (reference_contexts or []) if str(item).strip()] or None,
        )
        metrics = [
            components["Faithfulness"](llm=llm),
            components["ResponseRelevancy"](llm=llm, embeddings=embeddings),
        ]
        if str(reference_answer or "").strip():
            metrics.extend(
                [
                    components["LLMContextPrecisionWithReference"](llm=llm),
                    components["LLMContextRecall"](llm=llm),
                ]
            )

        scores: dict[str, float] = {}
        errors: dict[str, str] = {}
        for metric in metrics:
            metric_name = normalized_metric_name(str(metric.name or type(metric).__name__))
            try:
                metric.init(run_config)
                score = float(metric.single_turn_score(sample))
                if math.isfinite(score):
                    scores[metric_name] = round(score, 4)
                else:
                    errors[metric_name] = "non_finite_score"
            except Exception as exc:
                errors[metric_name] = f"{type(exc).__name__}: {str(exc)[:240]}"
        status = {
            "provider": "ragas",
            "version": components["version"],
            "status": "success" if scores and not errors else "partial" if scores else "failed",
            "evaluated_metrics": list(scores),
            "errors": errors,
        }
        return RagasCaseResult(scores=scores, status=status)


class LangChainEmbeddingAdapter:
    """Expose the project's embedding provider through LangChain's Embeddings contract."""

    def __new__(cls, provider: Any):
        from langchain_core.embeddings import Embeddings

        class Adapter(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return provider.embed_documents(texts)

            def embed_query(self, text: str) -> list[float]:
                return provider.embed_query(text)

        return Adapter()


def create_ragas_evaluator(workflow: Any, settings: Any) -> tuple[RagasEvaluator | None, str]:
    """Build an evaluator without importing Ragas during API startup."""
    model = getattr(getattr(workflow, "agent", None), "model", None)
    embedding_provider = getattr(getattr(workflow, "rag_engine", None), "embedding_provider", None)
    if model is None:
        return None, "agent_model_unavailable"
    if embedding_provider is None:
        return None, "embedding_provider_unavailable"
    try:
        return (
            RagasEvaluator(
                model=model,
                embedding_provider=embedding_provider,
                timeout_seconds=int(getattr(settings, "ragas_timeout_seconds", 60) or 60),
                max_retries=int(getattr(settings, "ragas_max_retries", 1) or 1),
            ),
            "",
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:240]}"


def load_ragas_components() -> dict[str, Any]:
    """Load Ragas lazily so online chat startup does not depend on it."""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    import ragas
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
    from ragas.run_config import RunConfig

    return {
        "version": str(getattr(ragas, "__version__", "")),
        "SingleTurnSample": SingleTurnSample,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "Faithfulness": Faithfulness,
        "ResponseRelevancy": ResponseRelevancy,
        "LLMContextPrecisionWithReference": LLMContextPrecisionWithReference,
        "LLMContextRecall": LLMContextRecall,
        "RunConfig": RunConfig,
    }


def normalized_metric_name(value: str) -> str:
    aliases = {
        "answer_relevancy": "answer_relevancy",
        "llm_context_precision_with_reference": "context_precision",
        "context_recall": "context_recall",
        "faithfulness": "faithfulness",
    }
    normalized = str(value or "").strip().lower()
    return aliases.get(normalized, normalized)

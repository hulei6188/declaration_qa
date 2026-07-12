"""Agentic RAG 工作流的请求预处理节点。"""

from __future__ import annotations

from pathlib import Path

from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.helpers.events import add_event
from declaration_qa.application.helpers.text_safety import mask_sensitive_text
from declaration_qa.rag.engine.text import normalize_text


class PreprocessingNodes:
    """只处理请求与查询图片，不读取任何工单业务数据。"""

    def context_build(self, state: DeclarationQAState) -> DeclarationQAState:
        """构造请求上下文，不加载或绑定任何工单。"""
        next_state = dict(state)
        next_state["region_code"] = next_state.get("region_code") or "340000"
        add_event(
            next_state,
            "context_built",
            {
                "ticket_context_bound": False,
                "recent_message_count": len(next_state.get("session_messages") or []),
            },
        )
        return next_state

    def query_image_enrich(self, state: DeclarationQAState) -> DeclarationQAState:
        """对用户查询图片执行 OCR，供知识检索工具使用。"""
        next_state = dict(state)
        query_image = next_state.get("query_image") or {}
        if not isinstance(query_image, dict) or not query_image:
            next_state["query_image_ocr_text"] = ""
            next_state["query_image_ocr_status"] = {"status": "not_applicable", "char_count": 0}
            return next_state

        client = getattr(self, "query_image_ocr_client", None)
        image_path = Path(str(query_image.get("file_path") or "").strip())
        if client is None or not image_path.is_file():
            status = "disabled" if client is None else "missing_file"
            next_state["query_image_ocr_text"] = ""
            next_state["query_image_ocr_status"] = {"status": status, "char_count": 0}
            add_event(next_state, "query_image_ocr", next_state["query_image_ocr_status"])
            return next_state

        max_chars = max(1, int(getattr(self, "query_image_ocr_max_chars", 4000) or 4000))
        cache_path = image_path.with_suffix(f"{image_path.suffix}.ocr.txt")
        cache_hit = False
        try:
            if cache_path.is_file():
                raw_ocr_text = cache_path.read_text(encoding="utf-8")
                cache_hit = True
            else:
                raw_ocr_text = client.extract_text(
                    filename=str(query_image.get("filename") or image_path.name),
                    content_type=str(query_image.get("content_type") or "image/png"),
                    data=image_path.read_bytes(),
                )
                normalized = str(raw_ocr_text or "").strip()
                if normalized:
                    try:
                        cache_path.write_text(normalized, encoding="utf-8")
                    except OSError:
                        pass
            raw_ocr_text = str(raw_ocr_text or "").strip()
            ocr_text = raw_ocr_text[:max_chars]
            status_payload = {
                "status": "success" if ocr_text else "empty",
                "char_count": len(ocr_text),
                "cache_hit": cache_hit,
                "truncated": len(raw_ocr_text) > max_chars,
            }
        except Exception as exc:
            ocr_text = ""
            status_payload = {
                "status": "failed",
                "char_count": 0,
                "cache_hit": False,
                "error": type(exc).__name__,
            }

        next_state["query_image_ocr_text"] = ocr_text
        next_state["query_image_ocr_status"] = status_payload
        add_event(next_state, "query_image_ocr", status_payload)
        return next_state

    def query_preprocess(self, state: DeclarationQAState) -> DeclarationQAState:
        """清洗用户问题，不在服务端执行关键词分类。"""
        next_state = dict(state)
        query = normalize_text(mask_sensitive_text(next_state.get("query") or ""))
        next_state["normalized_query"] = query
        next_state["biz_type"] = next_state.get("request_biz_type")
        add_event(next_state, "query_preprocessed", {"normalized_query": query, "biz_type": next_state.get("biz_type")})
        return next_state

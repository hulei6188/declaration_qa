"""Conversion support for legacy Microsoft Word .doc files."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from declaration_qa.indexing.document_parsing.models import DocumentParseError


DOC_CONTENT_TYPE = "application/msword"
DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
LEGACY_DOC_CONTENT_TYPES = {
    DOC_CONTENT_TYPE,
    "application/vnd.ms-word",
    "application/x-msword",
    "application/cdfv2",
}


@dataclass(frozen=True)
class LegacyDocConversionResult:
    """A converted DOCX payload and the converter used to produce it."""

    filename: str
    content_type: str
    data: bytes
    converter_command: str


def is_legacy_word_document(filename: str, content_type: str) -> bool:
    """Return whether the upload is an old binary Word document."""
    suffix = Path(str(filename or "")).suffix.lower()
    normalized_type = str(content_type or "").strip().lower()
    if suffix == ".docx":
        return False
    return suffix == ".doc" or normalized_type in LEGACY_DOC_CONTENT_TYPES


def convert_legacy_doc_to_docx(
    *,
    filename: str,
    data: bytes,
    converter_command: str = "",
    timeout_seconds: int = 120,
) -> LegacyDocConversionResult:
    """Convert one legacy Word document with LibreOffice in headless mode."""
    if not data:
        raise DocumentParseError("Legacy Word document is empty")
    command = resolve_converter_command(converter_command)
    safe_name = Path(str(filename or "document.doc")).name
    if Path(safe_name).suffix.lower() != ".doc":
        safe_name = f"{Path(safe_name).stem or 'document'}.doc"

    with TemporaryDirectory(prefix="declaration-qa-doc-") as temp_dir:
        work_dir = Path(temp_dir)
        source_path = work_dir / safe_name
        source_path.write_bytes(data)
        args = [
            *command,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            str(work_dir),
            str(source_path),
        ]
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_seconds)),
                check=False,
            )
        except FileNotFoundError as exc:
            raise DocumentParseError(
                f"Legacy Word converter was not found: {command[0]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise DocumentParseError(
                f"Legacy Word conversion timed out after {max(1, int(timeout_seconds))} seconds"
            ) from exc

        output_path = work_dir / f"{source_path.stem}.docx"
        if completed.returncode != 0 or not output_path.is_file():
            details = " ".join(
                part.strip()
                for part in (completed.stdout or "", completed.stderr or "")
                if part.strip()
            )
            suffix = f": {details}" if details else ""
            raise DocumentParseError(
                f"LibreOffice failed to convert legacy Word document{suffix}"
            )
        converted = output_path.read_bytes()
        if not converted:
            raise DocumentParseError("LibreOffice produced an empty DOCX file")
        return LegacyDocConversionResult(
            filename=output_path.name,
            content_type=DOCX_CONTENT_TYPE,
            data=converted,
            converter_command=command[0],
        )


def resolve_converter_command(converter_command: str) -> list[str]:
    """Resolve an explicit converter path or an installed LibreOffice executable."""
    configured = str(converter_command or "").strip().strip('"')
    if configured:
        if Path(configured).is_file():
            return [configured]
        parsed = shlex.split(configured, posix=False)
        if parsed:
            return [part.strip('"') for part in parsed]
    discovered = shutil.which("soffice") or shutil.which("libreoffice")
    if discovered:
        return [discovered]
    raise DocumentParseError(
        "Legacy Word conversion requires LibreOffice; configure "
        "DECLARATION_QA_LEGACY_DOC_CONVERTER_COMMAND"
    )

"""Isolated declaration QA application.

This package intentionally does not import the existing AgentBase runtime.
Start it with: uvicorn declaration_qa.api.server:app --host 127.0.0.1 --port 8010
"""

__all__ = ["__version__"]

__version__ = "0.1.0"

"""Gov Service Agent 政务智办应用。

此包有意不导入既有 AgentBase 运行时。
启动命令：uvicorn declaration_qa.api.server.app:app --host 127.0.0.1 --port 8010
"""

__all__ = ["__version__"]

__version__ = "0.1.0"

"""Agent Tool Registry 及内置业务工具。"""

from declaration_qa.application.tools.knowledge import KnowledgeSearchInput, KnowledgeSearchTool, KnowledgeSubQuery
from declaration_qa.application.tools.registry import (
    AgentTool,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolRegistry,
)
from declaration_qa.application.tools.service_queries import (
    AppointmentBookTool,
    AppointmentCancelTool,
    AppointmentHallQueryTool,
    AppointmentListQueryTool,
    AppointmentSlotQueryTool,
    ConsultationRecordQueryTool,
    DeliveryStatusQueryTool,
    PaymentOrderQueryTool,
    PaymentReceiptQueryTool,
    QueueStatusQueryTool,
    ResultCertificateQueryTool,
    SubmissionReceiptQueryTool,
)
from declaration_qa.application.tools.ticket import TicketDetailQueryTool, TicketListQueryTool

__all__ = [
    "AgentTool",
    "AppointmentBookTool",
    "AppointmentCancelTool",
    "AppointmentHallQueryTool",
    "AppointmentListQueryTool",
    "AppointmentSlotQueryTool",
    "ConsultationRecordQueryTool",
    "DeliveryStatusQueryTool",
    "KnowledgeSearchInput",
    "KnowledgeSearchTool",
    "KnowledgeSubQuery",
    "PaymentOrderQueryTool",
    "PaymentReceiptQueryTool",
    "QueueStatusQueryTool",
    "ResultCertificateQueryTool",
    "SubmissionReceiptQueryTool",
    "TicketDetailQueryTool",
    "TicketListQueryTool",
    "ToolExecutionContext",
    "ToolExecutionResult",
    "ToolRegistry",
]

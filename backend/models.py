import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Enum, DateTime
from sqlalchemy.dialects.postgresql import UUID

from database import Base


# --------------- Python enums mirroring the DB enums ---------------

class TicketStatus(str, enum.Enum):
    NEW = "New"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    CLOSED = "Closed"


class TicketPriority(str, enum.Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class TicketCategory(str, enum.Enum):
    FRAUD = "Fraud"
    PAYMENT_ISSUE = "Payment Issue"
    GENERAL = "General"


# --------------- ORM Model ---------------

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    customer_name = Column(String(255), nullable=False)
    email_body = Column(Text, nullable=False)
    status = Column(
        Enum(TicketStatus, name="ticket_status", create_constraint=True),
        nullable=False,
        default=TicketStatus.NEW,
    )
    priority = Column(
        Enum(TicketPriority, name="ticket_priority", create_constraint=True),
        nullable=False,
        default=TicketPriority.MEDIUM,
    )
    category = Column(
        Enum(TicketCategory, name="ticket_category", create_constraint=True),
        nullable=False,
        default=TicketCategory.GENERAL,
    )
    sentiment = Column(String(50), nullable=True)
    intent = Column(String(500), nullable=True)
    summary = Column(Text, nullable=True)
    transaction_id = Column(String(100), nullable=True)
    amount = Column(String(50), nullable=True)
    draft_response = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self):
        return (
            f"<Ticket(id={self.id}, customer='{self.customer_name}', "
            f"status='{self.status}', priority='{self.priority}', "
            f"category='{self.category}')>"
        )

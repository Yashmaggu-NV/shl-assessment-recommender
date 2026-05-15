"""
Pydantic schemas for the SHL Assessment Recommender API.
Defines strict request/response models for the stateless /chat endpoint.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """A single conversation message from user or assistant."""

    role: str = Field(..., description="Role of the message sender: 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant", "system"):
            raise ValueError("role must be 'user', 'assistant', or 'system'")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("content must not be empty")
        return v.strip()


class ChatRequest(BaseModel):
    """Request body for the POST /chat endpoint. Contains the full conversation history."""

    messages: List[Message] = Field(
        ...,
        min_length=1,
        description="Full conversation history. Every request must contain the complete history.",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("messages must contain at least one message")
        # Last message must be from user
        if v[-1].role != "user":
            raise ValueError("The last message must be from 'user'")
        return v


class Recommendation(BaseModel):
    """A single SHL assessment recommendation."""

    name: str = Field(..., description="Name of the SHL assessment from catalog")
    url: str = Field(..., description="Catalog URL for the assessment")
    test_type: str = Field(
        ...,
        description="Single-letter test type code(s): K, A, P, B, S, C, D, E",
    )


class ChatResponse(BaseModel):
    """Response body for the POST /chat endpoint."""

    reply: str = Field(..., description="Natural language reply from the agent")
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="List of 0–10 SHL assessments. Empty if clarifying or refusing.",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the conversation complete.",
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(default="ok")

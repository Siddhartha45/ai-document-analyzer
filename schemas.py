from pydantic import BaseModel, Field
from enum import Enum


class EntityType(str, Enum):
    person = "person"
    organization = "organization"
    location = "location"
    date = "date"
    other = "other"


class Entity(BaseModel):
    name: str
    type: EntityType


class ActionItem(BaseModel):
    task: str
    assignee: str | None = None
    deadline: str | None = None


class DocumentAnalysis(BaseModel):
    """llms output schema"""

    summary: str
    key_points: list[str]
    entities: list[Entity]
    action_items: list[ActionItem]


class DocumentRequest(BaseModel):
    """incoming request schema for documents"""

    text: str = Field(min_length=1)
    title: str = Field(min_length=1)


class QuestionRequest(BaseModel):
    """incoming request schema for rag"""

    question: str = Field(min_length=1)


class Operation(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


class CalculatorArgs(BaseModel):
    a: float
    b: float
    operation: Operation

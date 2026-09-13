from sqlalchemy import JSON, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from database import Base


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(Text, default="Untitled Document")
    text: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    key_points: Mapped[list] = mapped_column(JSON)
    entities: Mapped[list] = mapped_column(JSON)
    action_items: Mapped[list] = mapped_column(JSON)


class ChunkModel(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    chunk_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(3072))

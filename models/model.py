from sqlmodel import SQLModel, Field
from typing import Optional, List
from models.mixins import TimeMixin


class User(SQLModel, TimeMixin, table=True):
    __tablename__ = "user"
    id: Optional[str] = Field(default=None, primary_key=True)
    description: str = Field(default=None, nullable=False)
    type: str = Field(default=None, nullable=False)
    feature: List[str] = Field(default=None, nullable=True)
    status: str = Field(default="active", nullable=False)
    key: Optional[str] = Field(default=None, nullable=False)
    username: str = Field(default=None, nullable=False)
    password: str = Field(default=None, nullable=False)



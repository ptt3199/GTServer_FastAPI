from fastapi import HTTPException
import logging
from typing import TypeVar, Optional
from pydantic import BaseModel, validator
from sqlmodel import Enum
from datetime import datetime
from typing import List

T = TypeVar('T')


class Sex(str, Enum):
    MALE = "MALE"
    FEMALE = "MALE"
    OTHERS = "OTHERS"


# get root logger
logger = logging.getLogger(__name__)


class RegisterSchema(BaseModel):
    username: str
    email: str
    password: str
    company: str
    #phone_number: str
    #birth: str
    #sex: Sex

    # @validator("sex")
    # def sex_validation(cls, v):
    #     if hasattr(Sex, v) is False:
    #         raise HTTPException(status_code=400, detail="Invalid input sex")
    #     return v


class PendingUser(BaseModel):
    username: str
    email: str
    created_at: datetime


class PendingUserList(BaseModel):
    user_list: List[PendingUser]


class LoginSchema(BaseModel):
    username: str
    password: str


class AUpdateUSchema(BaseModel):
    username: str
    state: str
    company: Optional[T] = None


class RoleSchema(BaseModel):
    username: str
    role: str


class ForgotPasswordSchema(BaseModel):
    email: str
    new_password: str


class DetailSchema(BaseModel):
    status: str
    message: str
    result: Optional[T] = None


class ResponseSchema(BaseModel):
    detail: str
    username: str
    user_id: str
    company: str
    role: str
    result: Optional[T] = None


class ResponseRegisterSchema(BaseModel):
    detail: str



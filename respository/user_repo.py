from models.model import User
from sqlalchemy import update as sql_update
from sqlalchemy import text
from sqlalchemy.future import select
from sqlalchemy.sql import func
from respository.base_repo import BaseRepo
from config import db, commit_rollback


class UsersRepository(BaseRepo):
    model = User

    @staticmethod
    async def find_by_username(username: str):
        query = select(User).where(User.username == username)
        return (await db.execute(query)).scalar_one_or_none()

    @staticmethod
    async def find_by_email(email: str):
        query = select(User).where(User.email == email)
        return (await db.execute(query)).scalar_one_or_none()

    @staticmethod
    async def update_password(email: str, password: str):
        query = sql_update(User).where(User.email == email).values(
            password=password).execution_options(synchronize_session="fetch")
        await db.execute(query)
        await commit_rollback()

    @staticmethod
    async def update_username(email: str, username: str):
        query = sql_update(User).where(User.email == email).values(
            username=username).execution_options(synchronize_session="fetch")
        await db.execute(query)
        await commit_rollback()

    @staticmethod
    async def update_state(username: str, state: str):
        query = sql_update(User).where(User.username == username).values(
            state=state).execution_options(synchronize_session="fetch")
        await db.execute(query)
        await commit_rollback()

    @staticmethod
    async def update_company(username: str, company: str):
        query = sql_update(User).where(User.username == username).values(
            company=company).execution_options(synchronize_session="fetch")
        await db.execute(query)
        await commit_rollback()

    @staticmethod
    async def update_role(username: str, role: str):
        query = sql_update(User).where(User.username == username).values(
            role=role).execution_options(synchronize_session="fetch")
        await db.execute(query)
        await commit_rollback()

    # @staticmethod
    # async def update_num_requests(username: str):
    #     query = sql_update(User).where(User.username == username).values(
    #         num_requests=func.coaleste(text('num_requests'), 0) + 1
    #     ).execution_options(synchronize_session="fetch")
    #     await db.execute(query)
    #     await commit_rollback()

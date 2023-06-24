from models.model import User
from config import db
from sqlalchemy.future import select


class UserService:

    @staticmethod
    async def get_user_profile_by_username(username: str):
        query = select(User.username,
                       User.email,
                       User.company, User.id).where(User.username == username)
        return(await db.execute(query)).mappings().one()

    @staticmethod
    async def get_user_list(role: str, state: str):
        query = select(User.username,
                       User.email,
                       User.created_at).where(User.role==role, User.state==state)
        return (await db.execute(query)).mappings().all()

    @staticmethod
    async def get_user_list_by_state(state: str):
        query = select(User.username,
                       User.email,
                       User.created_at).where(User.state == state)
        return (await db.execute(query)).mappings().all()

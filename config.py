import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlalchemy.ext.declarative import declarative_base
from models.model import *

DB_CONFIG = f"postgresql+asyncpg://postgres:qpfiev95@localhost:8080/gt_db_2"


SECRET_KEY = "secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300


class AsyncDatabaseSession:


    def __init__(self) -> None:
        self.session = None
        self.engine = None


    def __getattr__(self,name):
        return getattr(self.session,name)


    def init(self):
        self.engine = create_async_engine(DB_CONFIG, future=True, echo=True)
        self.session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)()
        #self.engine = sqlalchemy.create_engine(DB_CONFIG)
        #self.session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        #self.base = declarative_base()


    async def create_all(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def create_table(self, table_name):
        pass

db = AsyncDatabaseSession()


async def commit_rollback():
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import db
from controller import users, dl_model2, dl_model3
import concurrent.futures
import asyncio

origins = [
    "http://localhost:8888"
]


def init_app():
    db.init()

    app = FastAPI(
        title="GT server",
        description="Admin page",
        version="1"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # start the app
    @app.on_event("startup")
    async def startup():
        await db.create_all()

    @app.on_event("shutdown")
    async def shutdown():
        await db.close()

    # app.include_router(users.router)
    app.include_router(dl_model2.router)
    return app


app = init_app()
if __name__ == '__main__':
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)

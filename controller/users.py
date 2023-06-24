from fastapi import APIRouter, Depends, Security
from schema.users import ResponseSchema, LoginSchema
from respository.user_repo import UsersRepository


router = APIRouter(prefix="", tags=['user'])


@router.post("/user")
async def get_user_profile(package: LoginSchema):
    username = package.username
    try:
        user = await UsersRepository.find_by_username(username)
    except:
        print("Cannot find user")
    return ResponseSchema(detail="Successfully fetch data!",
                          username=username, user_id="sadfasdf", company="asdfasdf", role="sadfasd")
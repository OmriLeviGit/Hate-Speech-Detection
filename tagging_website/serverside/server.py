import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from asyncio.locks import Lock

from pyexpat import features

from helper_functions.load_params import get_params

import controller
import db_service
from utils.base_models import Password, User_Id, Classification
from auth import login_required


app = FastAPI()
db = db_service.get_instance()
lock = Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)


@app.post("/auth/signin")
async def sign_in(data: Password):
    return await controller.handle_sign_in(data.password)


@app.get("/get_tweet_to_tag")
@login_required
async def get_tweet_to_tag(user_id):
    return await controller.get_tweet_to_tag(lock, user_id)


@app.post("/submit_tweet_tag")
@login_required
async def tag_tweet(user_id, data: Classification):
    tweet_id=data.tweet_id
    classification=data.classification
    features=data.features
    await controller.handle_tweet_tagging(lock, user_id, tweet_id, classification, features)


@app.get("/count_tags_made_by_user")
@login_required
async def count_tags_made(user_id):
    return await controller.count_tags_made(user_id)


@app.get("/get_user_panel")
@login_required
async def get_user_panel(user_id):
    return await controller.get_user_panel(user_id, lock)


@app.get("/get_pro_panel")
@login_required
async def get_pro_panel():
    return controller.get_pro_panel(lock)


@app.get("/params_list")
async def params_list():
    return get_params()


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
    # uvicorn.run("server:app", host="localhost", port=8000, reload=True)
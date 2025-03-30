import os
import platform
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from asyncio.locks import Lock

from pyexpat import features

from helper_functions.load_params import get_params

import controller
from base_models import Password, Classification
from auth import login_required

# update the time form utc to jerusalem or something

app = FastAPI()
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
    tagging_duration=data.tagging_duration
    await controller.handle_tweet_tagging(lock, user_id, tweet_id, classification, features, tagging_duration)

# Currently, not in use by the client
@app.get("/count_tags_made_by_user")
@login_required
async def count_tags_made(user_id):
    return await controller.count_tags_made(user_id)

@app.get("/tweets_left_to_classify")
@login_required
async def tweets_left_to_classify(user_id):
    return await controller.tweets_left_to_classify(user_id)


@app.get("/get_user_panel")
@login_required
async def get_user_panel(user_id):
    return await controller.get_user_panel(lock, user_id)


@app.get("/get_pro_panel")
@login_required
async def get_pro_panel(user_id):
    return await controller.get_pro_panel(lock)


@app.get("/features_list")
async def params_list():
    return get_params()


if __name__ == '__main__':
    if platform.system() == 'Linux':
        # otherwise, opens with docker
        print("Starting postgres..")
        os.system('sudo service postgresql start')

    host_address = "0.0.0.0" if os.path.exists('/.dockerenv') else "localhost"
    uvicorn.run("server:app", host=host_address, port=8080, reload=True)
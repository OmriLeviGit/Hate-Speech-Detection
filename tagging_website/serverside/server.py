import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from asyncio.locks import Lock
from helper_functions.load_params import get_params

import controller
import db_service
from serverside.utils.base_models import Password, User_Id, Classification
from auth import login_required

app = FastAPI()
db = db_service.get_instance()
lock = Lock()

"""
ours            theirs
User            Passcode
User.user_id    Passcode.id
User.password   Passcode.key
"""

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


@app.post("/classify_tweet")
# @login_required
async def classify_tweet(user_id, data: Classification):
    tweet_id, classification, reasons = data

    return await controller.handle_classification(lock, user_id, tweet_id, classification, reasons)


# TODO probably needs a basemodel class but im not sure, should work is it is though
@app.get("/count_classifications")
# @login_required
async def count_classifications(data): # TODO what type is this data and why is it not data.password
    return await controller.count_classifications(data)


@app.get("/get_user_panel")
# @login_required
async def get_user_panel(user_id):
    return await controller.get_user_panel(user_id, lock)


@app.get("/get_pro_panel")
# @login_required
async def get_pro_panel():
    return controller.get_pro_panel(lock)


@app.get("/params_list")
async def params_list():
    return get_params()


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
    # uvicorn.run("server:app", host="localhost", port=8000, reload=True)
from typing import List

from pydantic import BaseModel

class Password(BaseModel):
    password: str
    
# ToDo - User_Id might be unnecessary
class User_Id(BaseModel):
    user_id: str

class Classification(BaseModel):
    tweet_id: str
    classification: str
    features: List[str]
    tagging_duration: float
from pydantic import BaseModel

class Password(BaseModel):
    password: str
    
class User_Id(BaseModel):
    user_id: str

class Classification(BaseModel):
    tweet_id: str
    classification: str
    reasons: str
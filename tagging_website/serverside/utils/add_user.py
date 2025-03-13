from secrets import token_urlsafe
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_service import get_db_instance

script_dir = os.path.dirname(os.path.abspath(__file__))

def create_user_list(path, tweets_left, days_left):
    df = pd.read_excel(path)
    df.columns = df.columns.str.lower()
    emails = df["email"].tolist()
   
    users = []
   
    for email in emails:
        password = token_urlsafe(6)
        users.append([email, password, tweets_left, days_left])
    
    return users

if __name__ == "__main__":
    path = os.path.join(script_dir, "..", "data", "user_details.xlsx")
    tweets_left = 200
    days_left = 30
    
    user_list = create_user_list(path, tweets_left, days_left)
    db = get_db_instance()
    for user in user_list:
        db.create_user(*user)
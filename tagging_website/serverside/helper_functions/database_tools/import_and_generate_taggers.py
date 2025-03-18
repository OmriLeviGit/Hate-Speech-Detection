from datetime import datetime, timezone, timedelta
import pandas as pd
import sys
import os
import random
import string

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(script_dir)))
from db_service import get_db_instance


def generate_alphanumeric_password(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def create_user_list(path, due_date, tweets_left):
    df = pd.read_excel(path)
    df.columns = df.columns.str.lower()
    
    df["is pro"] = df["is pro"].astype(int).astype(bool)
    data = df[["email", "is pro"]].values.tolist()

    user_list = []
   
    for email, is_pro in data:
        password = email    # kept separately in case we want to change in the future

        if email in ["qwe@gmail.com", "asd@gmail.com", "zxc@gmail.com"]:
            password = email[:email.index('@')]

        if is_pro:
            entry = [email.lower(), password.lower(), None, 0, is_pro]
        else:
            entry = [email.lower(), password.lower(), due_date, tweets_left, is_pro]
            

        user_list.append(entry)
    
    return user_list


def generate_users(days_left, tweets_left, file_name="taggers_details.xlsx"):
    path = os.path.join(script_dir, "..", "..", "data", file_name)
    due_date = datetime.now() + timedelta(days=days_left) # TODO make sure the due date itself is allowed
    user_list = create_user_list(path, due_date, tweets_left)

    db = get_db_instance()
    for user in user_list:
        db.create_user(*user)


if __name__ == "__main__":
    file_name = "taggers_details.xlsx"
    days_left = 30
    tweets_left = 10

    generate_users(days_left, tweets_left, file_name)

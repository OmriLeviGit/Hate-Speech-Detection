import pandas as pd
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db_service

def import_tweets_from_csv(path="../ready_to_load/tweet_table.csv", limit=None):
    db = db_service.get_db_instance()

    df = pd.read_csv(path)

    for _, row in df.iterrows():
        if limit and limit <= i:
            break

        # Parse JSON fields
        photos_list = json.loads(row['photos']) if pd.notna(row['photos']) and row['photos'] else []
        
        # Extract photo URLs from the photos field
        photo_urls = []
        if photos_list:
            for photo in photos_list:
                if isinstance(photo, dict) and 'url' in photo:
                    photo_urls.append(photo['url'])
                elif isinstance(photo, str):
                    photo_urls.append(photo)
        
        # Parse tagged users
        tagged_users_list = []
        if pd.notna(row['tagged_users']) and row['tagged_users']:
            try:
                tagged_data = json.loads(row['tagged_users'])
                if isinstance(tagged_data, list):
                    for user in tagged_data:
                        if isinstance(user, dict) and 'url' in user:
                            tagged_users_list.append(user['url'])
            except json.JSONDecodeError:
                print(f"Error parsing tagged_users for tweet {row['url']}")
        
        # Parse hashtags
        hashtags_list = []
        if pd.notna(row['hashtags']) and row['hashtags']:
            try:
                hashtags_data = json.loads(row['hashtags'])
                if isinstance(hashtags_data, list):
                    hashtags_list = hashtags_data
            except json.JSONDecodeError:
                print(f"Error parsing hashtags for tweet {row['url']}")
        
        # Parse date
        tweet_date = None
        if pd.notna(row['date_posted']):
            try:
                tweet_date = datetime.fromisoformat(row['date_posted'].replace('Z', '+00:00'))
            except ValueError:
                print(f"Error parsing date for tweet {row['url']}")
        
        db.insert_tweet(
            tweet_id=row['id'],
            user_posted=row['user_posted'] if pd.notna(row['user_posted']) else None,
            content=row['description'] if pd.notna(row['description']) else "",
            date_posted=tweet_date,
            photos=photo_urls,
            tweet_url=row['url'],
            tagged_users=tagged_users_list,
            replies=int(row['replies']) if pd.notna(row['replies']) else 0,
            reposts=int(row['reposts']) if pd.notna(row['reposts']) else 0,
            likes=int(row['likes']) if pd.notna(row['likes']) else 0,
            views=int(row['views']) if pd.notna(row['views']) else 0,
            hashtags=hashtags_list
        )

        print(f"Added tweet: {row['url']}")



if __name__ == "__main__":
    path = "../ready_to_load/tweet_table.csv"
    import_tweets_from_csv(path)
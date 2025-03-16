import os
import sys
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
server_side = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(os.path.dirname(os.path.dirname(script_dir)))

from db_service import get_db_instance


def extract_data_of_interest(tagged_users_filename="tagged_users.csv", hashtags_filename="hashtags.csv"):
    db = get_db_instance()

    tagged_users_data = db.get_tagged_users()
    hashtags_data = db.get_hashtags()
    
    tagged_users_df = pd.DataFrame(tagged_users_data, columns=['tagged_user', 'count'])
    hashtags_df = pd.DataFrame(hashtags_data, columns=['hashtag', 'count'])

    tagged_users_path = os.path.join(server_side, "data", "data_of_interest", tagged_users_filename)
    hashtags_path = os.path.join(server_side, "data", "data_of_interest", hashtags_filename)
    
    tagged_users_df.to_csv(tagged_users_path, index=False)
    hashtags_df.to_csv(hashtags_path, index=False)
    
    return f"Data exported to {tagged_users_filename} and {hashtags_filename}"

if __name__ == "__main__":
    extract_data_of_interest()
    
    
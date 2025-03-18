import os
import sys
import random
import pandas as pd


script_dir = os.path.dirname(os.path.abspath(__file__))
server_side = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(os.path.dirname(os.path.dirname(script_dir)))

from db_service import get_db_instance

def filter_non_posters(usernames_with_counts, posters):
   non_posters = []
   for username_entry in usernames_with_counts:
       username = username_entry[0]
       if username not in posters:
           non_posters.append(username_entry)
   return non_posters


def extract_usernames(url_repetitions_tuples):
   usernames = []
   for url_tuple in url_repetitions_tuples:
       url = url_tuple[0]
       repetitions = url_tuple[1]
       # Extract username from the URL (everything after the last '/')
       username = url.split('/')[-1]
       usernames.append([username, repetitions])
   return usernames


def extract_tagged_users(filename):
    db = get_db_instance()

    posters = db.get_all_posters()
    posters = set([poster[0] for poster in posters])

    tagged_users = db.get_tagged_users()
    extracted_usernames = extract_usernames(tagged_users)

    non_posters = filter_non_posters(extracted_usernames, posters)
    
    tagged_users_df = pd.DataFrame(non_posters, columns=['tagged_user', 'count'])
    path = os.path.join(server_side, "data", "data_of_interest", filename)
    
    tagged_users_df.to_csv(path, index=False)
    
    return f"Tagged users data exported to {filename}"

def create_x_urls(usernames):
   x_urls = []
   
   for item in usernames:
       # Check if this is a [username, count] pair or just a username
       if isinstance(item, list) or isinstance(item, tuple):
           username = item[0]
           count = item[1]
           x_url = f"https://x.com/{username}"
           x_urls.append([x_url, count])
       else:
           # Just a username
           x_url = f"https://x.com/{item}"
           x_urls.append(x_url)
   
   return x_urls

def extract_and_randomize_tagged_users(input_filename, min_count=1):
    # Read the data
    input_path = os.path.join(server_side, "data", "data_of_interest", input_filename)
    df = pd.read_csv(input_path)

    # Extract tagged users and their counts
    tagged_users = []

    # Assuming the columns are named 'tagged_user' and 'count'
    # Adjust column names if they're different in your file
    for _, row in df.iterrows():
        if row['count'] >= min_count:
            tagged_users.append([row['tagged_user'], row['count']])

    # Randomize the list
    random.shuffle(tagged_users)

    x_urls_with_counts = create_x_urls(tagged_users)

    # Create output filename
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}_randomized.csv"
    output_path = os.path.join(server_side, "data", "data_of_interest", output_filename)

    # Save to new file
    randomized_df = pd.DataFrame(x_urls_with_counts, columns=['tagged_user', 'count'])
    randomized_df.to_csv(output_path, index=False)
      
def extract_hashtags(filename):
    db = get_db_instance()
    hashtags_data = db.get_hashtags()
    
    hashtags_df = pd.DataFrame(hashtags_data, columns=['hashtag', 'count'])
    path = os.path.join(server_side, "data", "data_of_interest", filename)
    
    hashtags_df.to_csv(path, index=False)
    
    return f"Hashtags data exported to {filename}"

if __name__ == "__main__":
    extract_tagged_users("tagged_users1.csv")
    extract_and_randomize_tagged_users("tagged_users1.csv", 2)

    extract_hashtags("hashtags.csv")
    
    
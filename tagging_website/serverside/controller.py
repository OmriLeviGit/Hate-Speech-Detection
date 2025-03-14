from datetime import datetime

from fastapi import HTTPException
from db_service import get_db_instance
from auth import generate_token
from types import SimpleNamespace

# ToDO - Check if each method that has to use the lock actually uses it

async def handle_sign_in(password):
    db = get_db_instance()

    user = db.get_user(password=password)

    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized: Could not find user")

    if not is_due_date_valid(user.user_id):
        raise HTTPException(status_code=401, detail="Unauthorized: due-date has passed")

    if not has_classifications_left(user.user_id):
        raise HTTPException(status_code=401, detail="Unauthorized: user does not have any classifications left")
        print(f"Error: A user with the password '{password}' does not exist")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # ToDo:
    #  - Understand if we need the is_valid method in the current project
    #  - If needed:
    #       - change the word 'key' to password, depending on how we set the database
    #       - Don't forget to use the left_to_classify
    # if not user.is_valid(db.get_num_classifications(user.key)):
    #     raise HTTPException(status_code=401, detail="Unauthorized")

    db.update_last_login(user.user_id)
    token = generate_token(user.user_id)
    return {'token': token, 'is_pro': user.professional}


# Checks if the user's due_date is still valid (not expired)
def is_due_date_valid(user_id):
    db = get_instance()
    due_date = db.get_user_due_date(user_id)
    if due_date is None:
        return False
    return due_date > datetime.utcnow().date()


# Checks if the user has classifications left to perform
def has_classifications_left(user_id):
    db = get_instance()
    left_to_classify = db.get_user_left_to_classify(user_id)
    return left_to_classify is not None and left_to_classify > 0

# Returns a tweet to tag by a specific user.
async def get_tweet_to_tag(lock, user_id):
    db = get_db_instance()

    async with lock:
        tweet = db.assign_unclassified_tweet(user_id)

        return {
            'id': tweet.tweet_id,
            'content': tweet.content,
            'tweet_url': tweet.tweet_url
        }


# Handles a tweet tagging that was received and stores it in the right place (tagging_results for a pro user and taggers_decisions for regular users)
async def handle_tweet_tagging(lock, user_id, tweet_id, classification, features):
    db = get_instance()

    if not has_classifications_left(user_id):
        raise HTTPException(status_code=401, detail="Unauthorized")
    db = get_db_instance()
    is_pro = db.is_pro(user_id)

    async with lock:

        # added for consistency and removes entry from the assigned_tweet table (even if pro)
        db.insert_to_taggers_decisions(tweet_id, user_id, classification, features)

        # Check if the user is a pro user or a regular one, and handle the tagging accordingly
        is_pro = db.is_pro(user_id)

        if is_pro:
            db.insert_to_tagging_results(tweet_id, classification, features, "Pro user")
            return

        # Else, the tag was made by a regular user
        db.decrement_left_to_classify(user_id)

        # Check if the tweet has already been classified twice by two different regular users
        decisions = db.get_tweet_decisions(tweet_id)
        if len(decisions) < 2:
            return

        # Extract classifications
        first_classification = decisions[0]["classification"]
        second_classification = decisions[1]["classification"]

        # Check if both taggers agreed on tweet's sentiment, and is in {Positive, Negative, Irrelevant}
        if (first_classification == second_classification) and (first_classification !="Uncertain"):
            combined_features = list(set(decisions[0]["features"] + decisions[1]["features"]))
            db.insert_to_tagging_results(tweet_id, first_classification, combined_features, "User agreement")
            return

        # Else, assign to pro bank
        db.assign_tweet_pro(tweet_id)


# Checks how many classifications where made by a specific user
async def count_tags_made(user_id):
    db = get_db_instance()
    return {"count": db.count_tags_made(user_id)}


async def get_user_panel(user_id, lock):
    db = get_db_instance()

    async with lock:
        classified_count = db.get_num_classifications(user_id)
        positive_count = db.get_positive_classification_count(user_id)
        negative_count = db.get_negative_classification_count(user_id)
        irrelevant_count = db.get_irrelevant_classification_count(user_id)
        time_left = db.get_days_left_to_classify(user_id)
        num_remaining = db.get_number_of_tweets_left_to_classify(user_id)
        avg_time = db.get_average_classification_time(user_id)

    # Calculate average time in seconds (for demonstration purposes)
    if avg_time is not None:
        average_time_seconds = f"{avg_time:.2f}"
    else:
        average_time_seconds = "N/A"
    
    # TODO update names and match to the client
    if classified_count is not None:
        return {'total': classified_count,
                'pos': positive_count,
                'neg': negative_count,
                'time': time_left,
                'remain': num_remaining,
                'avg': average_time_seconds,
                'irr': irrelevant_count}

    else:
        return {'error': 'Error getting user data'}
    

async def get_pro_panel(lock): # TODO probably remove the lock
    users = []
    async with lock:
        db = get_db_instance()
        user_data = db.get_users()
        total_classifications = db.get_total_classifications()
        total_negatives = db.get_total_negative_classifications()
        total_positives = db.get_total_positive_classifications()
        total_irrelevant = db.get_total_irrelevant_classifications()

        for user in user_data:
            user_id = user.user_id
            email = user.email
            classification_count = db.get_num_classifications(user_id)
            positive_count = db.get_positive_classification_count(user_id)
            negative_count = db.get_negative_classification_count(user_id)
            irrelevant_count = db.get_irrelevant_classification_count(user_id)
            avg_time = db.get_average_classification_time(user_id)
            
            if classification_count is not None:

                # Calculate average time in seconds (for demonstration purposes)
                if avg_time is not None:
                    average_time_seconds = f"{avg_time:.2f}"
                else:
                    average_time_seconds = "N/A"

                # Append user data to the list
                users.append({
                    "email": email,
                    "personalClassifications": classification_count,
                    "positiveClassified": positive_count,
                    "negativeClassified": negative_count,
                    "averageTime": average_time_seconds,
                    "irrelevantClassified":irrelevant_count
                })
            else:
                # Handle error case if data retrieval fails for the user
                users.append({
                    "email": email,
                    "error": "Error getting user data"
                })

    return {"users": users,
            "total": total_classifications,
            "total_pos": total_positives,
            "total_neg": total_negatives,
            "total_irr": total_irrelevant}
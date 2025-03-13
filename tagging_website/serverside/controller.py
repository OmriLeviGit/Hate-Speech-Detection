from auth import generate_token
from fastapi import HTTPException
# from db_access import get_instance
from db_service import get_instance
from types import SimpleNamespace

async def handle_sign_in(password):
    db = get_instance()

    user = db.get_user(password=password)
    # The next commented line is just for a mock for a user, in case there's no DB to work with
    # user = SimpleNamespace(user_id="123", password="pass", key="something")
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # ToDo:
    #  - Understand if we need the is_valid method in the current project
    #  - If needed, change the word 'key' to password, depending on how we set the database
    # if not user.is_valid(db.get_num_classifications(user.key)):
    #     raise HTTPException(status_code=401, detail="Unauthorized")

    token = generate_token(user.user_id)
    return {'token': token, 'is_pro': user.professional}

# Returns a tweet to tag by a specific user.
async def get_tweet_to_tag(lock, user_id):
    db = get_instance()

    async with lock:
        user = db.get_user(user_id=user_id)
        # Check if the user already has a tweet assigned they need to tag
        if user and user.current_tweet_id:
            tweet = db.get_tweet(user.current_tweet_id)
        else:
            # If no assigned tweet, get a new one
            tweet = db.assign_unclassified_tweet(user_id)

    if tweet:
        return {
            'id': tweet.tweet_id,
            'content': tweet.content,
            'tweet_url': tweet.tweet_url
        }
    else:
        return {'error': 'No available tweets'}


# async def get_tweet_to_classify(lock, user_id):
#     """
#     check if user.user_id has a current_tweet_id.
#
#     if not:
#         list = ask the DB to get a list of all unclassified tweets (NOT IN PRO BANK and NOT IN final result and appears less than twice on taggers_decisions)
#         tweet_id = get a random tweet from list
#         Update for user_id the field of current_tweet_id to be tweet_id
#         ask the DB to get the tweet with tweet_id
#         tweet = get_tweet_by_id(tweet_id)
#
#     return {'id': tweet.id, 'content': tweet.content, 'tweet_url': tweet.tweet_url}
#     make sure the fields on return are the same as in the client
#
#     """


async def handle_classification(lock, user_id, tweet_id, classification, reasons):
    db = get_instance()
    if not db.get_passcode(user_id).is_valid(db.get_num_classifications(user_id)):
        raise HTTPException(status_code=401, detail="Unauthorized")
    tweet = db.get_tweet(tweet_id) # TODO maybe need await
    if tweet is not None:
        if classification.classification not in ['Positive', 'Negative', 'Irrelevant', 'Unknown']:
            return {'error': 'Invalid classification'}
        async with lock:
            result = db.classify_tweet(tweet_id, user_id, classification, reasons)
        return {'classified': result}
    else:
        return {'error': 'No such tweet'}


async def count_classifications(user_id): # TODO what is this data? whats the type
    db = get_instance()

    return {"count": db.get_num_classifications(user_id)}

async def get_user_panel(user_id, lock):
    db = get_instance()

    async with lock:
        classified_count = db.get_num_classifications(user_id)
        positive_count = db.get_num_positive_classifications(user_id)
        negative_count = db.get_num_negative_classifications(user_id)
        time_left = db.get_time_left(user_id)
        num_remaining = db.get_num_remaining_classifications(user_id)
        avg_time = db.get_average_classification_time(user_id)
        irrelevant_count = db.get_num_irrelevant_classifications(user_id)

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
        db = get_instance()
        user_data = db.get_users()
        total_classifications = db.get_total_classifications()
        total_negatives = db.get_total_negative_classifications()
        total_positives = db.get_total_positive_classifications()
        total_irrelevant = db.get_total_irrelevant_classifications()

        for user in user_data:
            password = user.key
            email = user.email
            classification_count = db.get_num_classifications(password)
            positive_count = db.get_num_positive_classifications(password)
            negative_count = db.get_num_negative_classifications(password)
            irrelevant_count = db.get_num_irrelevant_classifications(password)
            avg_time = db.get_average_classification_time(password)
            
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
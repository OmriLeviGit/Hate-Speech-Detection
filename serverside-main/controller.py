from auth import generate_token
from fastapi import HTTPException
from db_access import get_database_instance


async def handle_sign_in(password):
    db = get_database_instance()
    user = db.get_user(password)

    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not user.is_valid(db.get_num_classifications(user.key)): # TODO change the word 'key' to password, depending on how we set the database
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = generate_token(user.key)
    return {'token': token, 'is_pro': user.professional}


async def get_tweet(user_id, lock):
    db = get_database_instance()

    async with lock:
        tweet = db.get_unclassified_tweet(user_id)
        if tweet is not None:
            db.update_start(tweet, user_id)

    if tweet is not None:
        return {'id': tweet.id, 'tweeter': tweet.tweeter, 'content': tweet.content}
    else:
        return {'error': 'No unclassified tweets'}

    
async def handle_classification(lock, user_id, tweet_id, classification, reasons):
    db = get_database_instance()
    
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
    db = get_database_instance()

    return {"count": db.get_num_classifications(user_id)}

async def get_user_panel(user_id, lock):
    db = get_database_instance()

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
        db = get_database_instance()
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
from datetime import datetime

from fastapi import HTTPException
from db_service import get_db_instance
from auth import generate_token
from auth import generate_token
from types import SimpleNamespace

async def handle_sign_in(password):
    db = get_db_instance()

    user = db.get_user(password=password)

    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized: Could not find user")

    db.update_last_login(user.user_id)
    token = generate_token(user.user_id)
    return {'token': token, 'is_pro': user.professional}


# Checks if the user's due_date is still valid (not expired)
def is_due_date_valid(user_id):
    db = get_db_instance()
    due_date = db.get_user_due_date(user_id)
    if due_date and due_date >= datetime.utcnow().date():
        return True
    return False


async def get_tweet_to_tag(user_id):
    """
    Get next tweet for user to classify, with validation checks.

    Args:
        user_id: The ID of the user requesting a tweet

    Returns:
        dict: Tweet data or error response
    """
    db = get_db_instance()

    # Check if user's due date has passed (unless they're pro)
    if not is_due_date_valid(user_id) and not db.is_pro(user_id):
        return {'error': 'DEADLINE_PASSED', 'message': 'Due date has passed.'}

    # Check if user has any tweets left to classify
    tweets_left = db.tweets_left_to_classify(user_id)
    if tweets_left < 1:
        return {'error': 'NO_TWEETS_REMAINING', 'message': 'No tweets left to classify! 🎉'}

    # Try to get or assign a tweet to the user
    tweet_data = db.get_or_assign_tweet(user_id)
    if not tweet_data:
        return {'error': 'NO_AVAILABLE_TWEETS', 'message': 'No available tweets'}

    # Success - return tweet data
    return tweet_data


# Returns the number of tweets left to classify for a specific user.
# Don't confuse it with controller.has_classification_left
# The current method is different by sending the data to the client
async def tweets_left_to_classify(user_id):
    db = get_db_instance()
    return {"tweets_left": db.tweets_left_to_classify(user_id)}


# Checks if the user has classifications left to perform
def has_classifications_left(user_id):
    db = get_db_instance()
    left_to_classify = db.tweets_left_to_classify(user_id)
    return left_to_classify is not None and left_to_classify > 0


# Handles a tweet tagging that was received and stores it in the right place (tagging_results for a pro user and taggers_decisions for regular users)
async def handle_tweet_tagging(user_id, tweet_id, classification, features, tagging_duration):
    if not has_classifications_left(user_id):
        raise HTTPException(status_code=401, detail="Unauthorized")

    db = get_db_instance()

    # Removes entry from the assigned_tweet table (even if pro) and adds to the taggers decisions table
    db.insert_to_taggers_decisions(tweet_id, user_id, classification, features, tagging_duration)
    db.decrement_left_to_classify(user_id)

    is_pro = db.is_pro(user_id)

    if is_pro:
        db.insert_to_tagging_results(tweet_id, classification, features, "Pro user")
        return

    # Check if the tweet has already been classified twice by two different regular users
    decisions = db.get_tweet_decisions(tweet_id)

    if len(decisions) < 2:
        return

    # Extract classifications
    first_classification = decisions[0]["classification"]
    second_classification = decisions[1]["classification"]

    # If one of the taggers is uncertain, or if the taggers do not agree, let a pro user decide
    if first_classification == "Uncertain" or second_classification == "Uncertain" or first_classification != second_classification:
        db.assign_tweet_to_pro(tweet_id)
        return

    # The taggers agree and are certain
    combined_features = list(set(decisions[0]["features"] + decisions[1]["features"]))
    db.insert_to_tagging_results(tweet_id, first_classification, combined_features, "User agreement")

# Checks how many classifications where made by a specific user
async def count_tags_made(user_id):
    db = get_db_instance()
    return {"count": db.count_tags_made(user_id)}


async def get_user_panel(user_id):
    db = get_db_instance()

    # Get all classification counts in a single database call to improve performance
    classification_counts = db.get_classification_counts_by_type(user_id)

    # Return error if we couldn't retrieve classification data
    if classification_counts is None:
        return {'error': 'Error getting user data'}

    # Fetch remaining user progress metrics
    time_left = db.get_days_left_to_classify(user_id)  # Days until deadline
    num_remaining = db.tweets_left_to_classify(user_id)  # Unclassified tweets count
    avg_time = db.get_average_classification_time(user_id)  # Average time per classification

    # Build response with all user panel data
    return {
        'total': sum(classification_counts.values()),  # Total classifications made
        'pos': classification_counts.get('positive', 0),  # Positive classifications
        'neg': classification_counts.get('negative', 0),  # Negative classifications
        'irr': classification_counts.get('irrelevant', 0),  # Irrelevant classifications
        'unc': classification_counts.get('uncertain', 0),  # Uncertain classifications
        'time': time_left,  # Days remaining
        'remain': num_remaining,  # Tweets left to classify
        'avg': avg_time,  # Average classification time
    }


async def get_pro_panel():
    users = []

    db = get_db_instance()
    user_data = db.get_users()

    total_classifications = db.get_total_classifications()
    total_negatives = db.get_total_negative_classifications()
    total_positives = db.get_total_positive_classifications()
    total_irrelevant = db.get_total_irrelevant_classifications()
    total_uncertain = db.get_total_uncertain_classifications()

    for user in user_data:

        user_id = user.user_id
        email = user.email

        classification_count = db.count_tags_made(user_id)
        left_to_classify = db.tweets_left_to_classify(user_id)
        positive_count = db.get_positive_classification_count(user_id)
        negative_count = db.get_negative_classification_count(user_id)
        irrelevant_count = db.get_irrelevant_classification_count(user_id)
        uncertain_count = db.get_uncertain_classification_count(user_id)
        avg_time = db.get_average_classification_time(user_id)

        if classification_count is not None:

            # Append user data to the list
            users.append({
                "email": email,
                "personalClassifications": classification_count,
                "leftToClassify": left_to_classify,
                "positiveClassified": positive_count,
                "negativeClassified": negative_count,
                "averageTime": avg_time,
                "irrelevantClassified":irrelevant_count,
                "uncertainClassified": uncertain_count,
            })
        else:
            # Handle error case if data retrieval fails for the user
            users.append({
                "email": email,
                "error": "Error getting user data"
            })

    return {
        "users": users,
        "total": total_classifications,
        "total_pos": total_positives,
        "total_neg": total_negatives,
        "total_irr": total_irrelevant,
        "total_unc": total_uncertain
    }
import random
from datetime import timedelta
from pprint import pprint
from secrets import token_urlsafe

import pandas as pd
from sqlalchemy import Engine, Nullable, func, text
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from credentials import *
from model import *
from serverside.helper_functions.tweets_helpers import fix_corrupted_text

class Singleton(type):
    def __init__(cls, name, bases, dict):
        super().__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super().__call__(*args, **kw)
        return cls.instance


class get_instance(metaclass=Singleton):
    def __init__(self):
        self.engine: Engine = create_engine(DB)

    # Creates a user and generates a password to them
    # ToDo: Take care of last login to be null when first created,
    def create_user(self, email, num_days, left_to_classify):

        password = token_urlsafe(6)
        due_date = datetime.now() + timedelta(days=num_days)

        user = (User(
                password=password,
                email = email,
                due_date = due_date,
                left_to_classify = left_to_classify))

        with Session(self.engine) as session:
            session.add(user)
            session.commit()

        return user

    # Returns a user object
    def get_user(self, user_id=None, password=None):
        with Session(self.engine) as session:
            if user_id:
                user = session.query(User).filter(User.user_id == user_id)
            elif password:
                user = session.query(User).filter(User.password == password)
            return user.one_or_none()

    # Returns all users from users table
    def get_users(self):
        with Session(self.engine) as session:
            return session.query(User).all()

    # Returns a tweet object
    def get_tweet(self, tweet_id):
        with Session(self.engine) as session:
            return session.query(Tweet).filter(Tweet.tweet_id == tweet_id).one_or_none()

    # Adding a new tweet into the tweets table in the DB
    def insert_tweet(self, tweet_id, user_posted, content, date_posted, photos, tweet_url, tagged_users, replies, reposts, likes, views, hashtags):

        if not content:
            return

        processed_content = fix_corrupted_text(content)

        tweet = (Tweet
                 (tweet_id=tweet_id,
                  user_posted=user_posted,
                  content=processed_content,
                  date_posted=date_posted,
                  photos=photos,
                  tweet_url=tweet_url,
                  tagged_users=tagged_users,
                  replies=replies,
                  reposts=reposts,
                  likes=likes,
                  views=views,
                  hashtags=hashtags))

        with Session(self.engine) as session:
            session.add(tweet)
            session.commit()

    # ToDo - the following method has yet to be tested and done
    def assign_unclassified_tweet(self, user):
        """
        Assign an unclassified tweet to the given user.
        A tweet is considered unclassified if:
        - It's NOT in tagging_results (not already finalized)
        - It's NOT in pro_bank (not reserved for professionals)
        - It appears less than 2 times in taggers_decisions (not assigned to 2 users yet)
        """

        with Session(self.engine) as session:
            # Step 1: Find all eligible tweets
            subquery = (
                session.query(
                    Tweet.tweet_id
                )
                .outerjoin(TaggingResult, Tweet.tweet_id == TaggingResult.tweet_id)
                .outerjoin(ProBank, Tweet.tweet_id == ProBank.tweet_id)
                .outerjoin(TaggersDecision, Tweet.tweet_id == TaggersDecision.tweet_id)
                .group_by(Tweet.tweet_id)
                .having(func.count(TaggersDecision.tagger_decision_id) < 2)  # Less than 2 assignments
                .filter(TaggingResult.id.is_(None))  # Not finalized
                .filter(ProBank.id.is_(None))  # Not reserved for professionals
                .order_by(func.random())  # Randomize selection
                .limit(1)
                .subquery()
            )

            # Step 2: Retrieve a random tweet from the eligible ones
            random_tweet = session.query(Tweet).filter(Tweet.tweet_id == subquery.c.tweet_id).first()

            if random_tweet:
                # Assign the tweet to the user
                user.current_tweet_id = random_tweet.tweet_id
                session.add(user)  # Ensure user update is tracked

                # Reserve the tweet in taggers_decisions table with "N/A"
                reservation = TaggersDecision(
                    tweet_id=random_tweet.tweet_id,  # Use tweet_id instead of object
                    tagged_by=user.user_id,
                    classification="N/A"
                )

                session.add(reservation)
                session.commit()  # Commit both user update & new reservation
                return random_tweet.tweet_id
            else:
                return None  # No available tweet


    # ToDo - Create def __reserve_tweet(self, tweet, passcode)

    # ToDo - Create def __reserve_tweet_pro(self, tweet, passcode, pro_tweet)

    # ToDo - Create def insert_to_pro_bank(self, tweet_id)

    # ToDo - Create def get_unclassified_tweet(self, passcode)

    # ToDo - Create def def classify_tweet(self, tweet_id, user_id, classification, features)

    # ToDo - Create def classify_tweet_pro(self, tweet_id, passcode, classification, features)

    # ToDo - Create def get_average_classification_time(self, classifier), a method that calculates duration of a tweet tag called

    # def get_tweet_to_classify(self, user_id):

    # Flow to get a tweet to classify:
    #   A user enters the client and asks for a tweet to tag
    #   check if the tweet IS NOT IN pro_bank and NOT IN tagging_results
    #

    # Returns the number of classifications marked as "Positive" made by a specific user
    def get_positive_classification_count(self, user_id):
        with Session(self.engine) as session:
            return session.query(TaggersDecision).filter(TaggersDecision.tagged_by == user_id).filter(
                TaggersDecision.tagged_by == "Positive").count()

        # Returns the number of classifications marked as "Negative" made by a specific user

    def get_negative_classification_count(self, user_id):
        with Session(self.engine) as session:
            return (
                session.query(TaggersDecision)
                .filter(TaggersDecision.tagged_by == user_id).filter(
                    TaggersDecision.tagged_by == "Negative")
                .count())

    # Returns the number of classifications marked as "Irrelevant" made by a specific user
    def get_irrelevant_classification_count(self, user_id):
        with Session(self.engine) as session:
            return (session.query(TaggersDecision)
                    .filter(TaggersDecision.tagged_by == user_id)
                    .filter(TaggersDecision.tagged_by == "Irrelevant").count())

    # ToDo - This method probably will change according to how we use taggers_decisions implementation
    # Return the total number of classification made by the taggers
    def get_total_classifications(self):
        with Session(self.engine) as session:
            return session.query(func.count(func.distinct(TaggersDecision.tweet_id))) \
                .filter(TaggersDecision.classification != "N/A") \
                .scalar()

    # ToDo - This method probably will change according to how we use taggers_decisions implementation
    # Return the total number of negatives classification made by the taggers
    def get_total_negative_classifications(self):
        with Session(self.engine) as session:
            return session.query(func.count(func.distinct(TaggersDecision.tweet_id))) \
                .filter(TaggersDecision.classification != "Negative") \
                .scalar()

    # ToDo - This method probably will change according to how we use taggers_decisions implementation
    # Return the total number of negatives classification made by the taggers
    def get_total_positive_classifications(self):
        with Session(self.engine) as session:
            return session.query(func.count(func.distinct(TaggersDecision.tweet_id))) \
                .filter(TaggersDecision.classification != "Positive") \
                .scalar()

    # ToDo - This method probably will change according to how we use taggers_decisions implementation
    # Return the total number of negatives classification made by the taggers
    def get_total_irrelevant_classifications(self):
        with Session(self.engine) as session:
            return session.query(func.count(func.distinct(TaggersDecision.tweet_id))) \
                .filter(TaggersDecision.classification != "Irrelevant") \
                .scalar()


    # Returns the number of classifications left for a specific user
    def get_number_of_tweets_left_to_classify(self, user_id):
        with Session(self.engine) as session:
            return (
                session.query(User.left_to_classify)
                .filter(User.user_id == user_id)
                .scalar()
            )

    # Return the number of days reminded until due date for a specific user
    def get_days_left_to_classify(self, user_id):
        with Session(self.engine) as session:
            current_date = datetime.now().date()
            user = session.query(User).filter(User.user_id == user_id).first()
            if user is not None and user.due_date is not None:
                days_left = (user.due_date - current_date).days
                return days_left if days_left >= 0 else 0
            else:
                return 0

    # ToDo - This method probably will change according to how we use taggers_decisions implementation
    # Returns the number of tweets classified by a user.
    def get_num_classifications(self, user_id):
        with Session(self.engine) as session:
            return session.query(TaggersDecision).filter(TaggersDecision.tagged_by == user_id).filter(
                TaggersDecision.classification != "N/A").count()





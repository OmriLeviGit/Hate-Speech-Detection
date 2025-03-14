import pytz
from sqlalchemy import Engine, Nullable, func, text
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from credentials import *
from model import *
from helper_functions.tweets_helpers import fix_corrupted_text
from datetime import datetime, timezone, timedelta
from secrets import token_urlsafe


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
    def create_user(self, email, num_days, left_to_classify):
        # ToDo - check if still necessary to generate a password here, maybe Omri does it on a different file
        password = token_urlsafe(6)
        due_date = datetime.now() + timedelta(days=num_days)
        user = (User(
                password=password,
                email=email,
                due_date=due_date,
                left_to_classify=left_to_classify))

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

    # Checks if a given user is a pro user (else, it's a regular user)
    def is_pro(self, user_id):
        with Session(self.engine) as session:
            user = session.query(User).filter(User.user_id == user_id).first()
            if user:
                return user.professional


    # Updates the last_login field of a user to the current timestamp
    def update_last_login(self, user_id):
        with Session(self.engine) as session:
            user = session.query(User).filter(User.user_id == user_id).first()
            if user:
                user.last_login = datetime.now(pytz.timezone('Asia/Jerusalem'))
                session.commit()


    # Decrements left_to_classify for a specific user
    def decrement_left_to_classify(self, user_id):
        with Session(self.engine) as session:
            user = session.query(User).filter(User.user_id == user_id).first()

            if not user or user.left_to_classify <= 0:
                return # Either user doesn't exist or has no classifications left

            user.left_to_classify -= 1  # Decrement count
            session.commit()


    # Returns the user's due_date from the database
    def get_user_due_date(self, user_id):
        with Session(self.engine) as session:
            user = session.query(User.due_date).filter(User.user_id == user_id).first()
            if user:
                return user.due_date
            return None


    # Returns the user's left_to_classify from the database
    def get_user_left_to_classify(self, user_id):
        with Session(self.engine) as session:
            user = session.query(User.left_to_classify).filter(User.user_id == user_id).first()
            if user:
                return user.left_to_classify
            return None


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


    # Assigns an unclassified tweet to the given user
    def assign_unclassified_tweet(self, user):
        """
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
                    tweet_id=str(random_tweet.tweet_id),
                    tagged_by=user.user_id,
                    classification="N/A"
                )
                session.add(reservation)
                session.commit()  # Commit both user update & new reservation
                return random_tweet.tweet_id

            else:
                return None  # No available tweet


    # Inserts a tagging decision to the taggers_decisions table
    def insert_to_taggers_decisions(self, tweet_id, user_id, tag_result, features):
        # Ensures tweet_id is a string to match the type on the DB
        tweet_id = str(tweet_id)

        with Session(self.engine) as session:
            # Check if there’s already a reservation entry to the (tweet_id, user_id) where (classification = "N/A")
            existing_record = (
                session.query(TaggersDecision)
                .filter(TaggersDecision.tweet_id == tweet_id, TaggersDecision.tagged_by == user_id)
                .filter(TaggersDecision.classification == "N/A")
                .first()
            )

            if existing_record:
                # If a reservation exists, update it with the actual classification
                existing_record.classification = tag_result
                existing_record.features = features if features else []
                existing_record.tagging_date = datetime.now(pytz.timezone('Asia/Jerusalem'))

            # Inserts new record into taggers_decisions
            else:
                new_entry = TaggersDecision(
                    tweet_id=tweet_id,
                    tagged_by=user_id,
                    classification=tag_result,
                    features=features if features else [],  # Ensure it's a list
                    tagging_date=datetime.now(pytz.timezone('Asia/Jerusalem')),
                    tagging_duration=None  # ToDo - Add the calculation of the tagging_duration
                )
                session.add(new_entry)

            session.commit()


    # Retrieves the last two classifications for a given tweet from the taggers_decisions table
    def get_tweet_decisions(self, tweet_id):
        tweet_id = str(tweet_id)  # Ensure tweet_id is a string

        with Session(self.engine) as session:
            decisions = (
                session.query(TaggersDecision.classification, TaggersDecision.features)
                .filter(TaggersDecision.tweet_id == tweet_id)
                .order_by(TaggersDecision.tagging_date.desc())  # Get latest decisions first
                .limit(2)  # Only retrieve the last two classifications
                .all()
            )

        # Convert to a list of dictionaries
        return [{"classification": d.classification, "features": d.features} for d in decisions]


    # Inserts a tagging result into the table
    def insert_to_tagging_results(self, tweet_id, tag_result, features, decision_source ):
        # Ensures tweet_id is a string to match the type in the DB
        tweet_id = str(tweet_id)

        with Session(self.engine) as session:
            # Check if tweet_id already exists in tagging_results
            exists = session.query(TaggingResult).filter(TaggingResult.tweet_id == tweet_id).first()

            if exists:
                return

            # Insert new record into tagging_results
            new_entry = TaggingResult(
                tweet_id=tweet_id,
                tag_result=tag_result,
                features=features if features else [],  # Ensure it's a list
                decision_source=decision_source
            )

            session.add(new_entry)
            session.commit()


    # ToDo - Make sure it assigns pro_bank tweet classifications evenly between pro users
    # Inserts a tweet into the pro_bank table if it's not already present
    def insert_tweet_to_pro_bank(self, tweet_id):
        with Session(self.engine) as session:
            # Cast tweet_id to match the DB type
            tweet_id = str(tweet_id)
            # Check if the tweet is already in pro_bank
            exists = session.query(ProBank).filter(ProBank.tweet_id == tweet_id).first()
            if exists:
                return False  # Tweet is already in pro_bank, no need to insert
            # Insert new record into pro_bank
            new_entry = ProBank(tweet_id=tweet_id)
            session.add(new_entry)
            session.commit()


    # Removes a tweet from pro_bank
    def remove_tweet_from_pro_bank(self, tweet_id):
        # Cast tweet_id to match the DB type
        tweet_id = str(tweet_id)
        with Session(self.engine) as session:
            record = session.query(ProBank).filter(ProBank.tweet_id == tweet_id).first()
            if record:
                session.delete(record)
                session.commit()

    # ToDo - Create def get_average_classification_time(self, classifier):
    #  - A method that calculates duration of a tweet tag called


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





import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from sqlalchemy import Engine, func
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from model import *
from helper_functions.tweets_helpers import fix_corrupted_text


load_dotenv('.env.local')
if os.path.exists('/.dockerenv'):
    # when running with docker, dont forget to stop the local post gres by using 'sudo service postgresql stop'
    DB = os.environ.get('DATABASE_DOCKER')
else:
    # if using the local machine, you need to start postgres by using 'sudo service postgresql start'
    DB = os.environ.get('DATABASE_LOCAL')

class Singleton(type):
    def __init__(cls, name, bases, dict):
        super().__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super().__call__(*args, **kw)
        return cls.instance


class get_db_instance(metaclass=Singleton):
    def __init__(self):
        self.engine: Engine = create_engine(DB)

    # Creates a user and generates a password to them
    # ToDo: Take care of last login to be null when first created,
    def create_user(self, email, password, due_date, tweets_left, is_pro):

        user = (User(
                password=password,
                email=email,
                due_date=due_date,
                left_to_classify=tweets_left,
                professional=is_pro
                ))

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
            else:
                print(f"error in db_service.get_user: {user_id}, {password}")
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
                user.last_login = datetime.now(timezone.utc)
                session.commit()

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

        processed_content = fix_corrupted_text(content)

        if not processed_content:   # if could not be parsed correctly
            return

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

        """
        new:
        A tweet is considered unclassified if:
        - It's NOT in tagging_results (not already finalized)
        - It's assigned to 1 user at most, and that user is not pro user
        """

        with Session(self.engine) as session:
            # Step 1: Find all eligible tweets
            assignment_info = (
                session.query(
                    Tweet.tweet_id,
                    func.count(AssignedTweet.user_id).label("assignment_count"),
                    func.bool_or(User.professional).label("has_pro_assignment")
                )
                .outerjoin(AssignedTweet, Tweet.tweet_id == AssignedTweet.tweet_id)
                .outerjoin(User, AssignedTweet.user_id == User.user_id)
                .group_by(Tweet.tweet_id)
                .subquery()
            )

            subquery = (
                session.query(Tweet.tweet_id)
                .outerjoin(TaggingResult, Tweet.tweet_id == TaggingResult.tweet_id)
                .join(assignment_info, Tweet.tweet_id == assignment_info.c.tweet_id, isouter=True)
                .filter(TaggingResult.id.is_(None))  # Not finalized
                .filter(
                    (assignment_info.c.assignment_count <= 1) & (assignment_info.c.has_pro_assignment == False)
                )
                .order_by(func.random())
                .limit(1)
                .subquery()
            )

            # Step 2: Retrieve a random tweet from the eligible ones
            random_tweet = session.query(Tweet).filter(Tweet.tweet_id == subquery.c.tweet_id).first()
            if random_tweet:
                # Assign the tweet to the user in the assigned_tweets table
                assignment = AssignedTweet(
                    user_id=user.user_id,
                    tweet_id=random_tweet.tweet_id,
                    completed=False
                )
                session.add(assignment)
                session.commit()
                return random_tweet.tweet_id
            else:
                return None  # No available tweet


    # Inserts a tagging decision to the taggers_decisions table
    def insert_to_taggers_decisions(self, tweet_id, user_id, tag_result, features):
        # Ensures tweet_id is a string to match the type on the DB
        tweet_id = str(tweet_id)

        # TODO as it curently is, tagging duration is always none

        with Session(self.engine) as session:
            # Inserts new record into taggers_decisions
            new_entry = TaggersDecision(
                tweet_id=tweet_id,
                tagged_by=user_id,
                classification=tag_result,
                features=features if features else [],  # Ensure it's a list
                tagging_date=datetime.now(timezone.utc),
                tagging_duration=None  # If you have the duration, pass it; otherwise, it's NULL
            )
            session.add(new_entry)

            # Remove from assigned_tweets table # TODO check if its called for pro users too
            assigned_tweet = session.query(AssignedTweet).filter(
                AssignedTweet.tweet_id == tweet_id,
                AssignedTweet.user_id == user_id
            ).first()

            if assigned_tweet:
                session.delete(assigned_tweet)

            session.commit()
            return True

    def get_tweet_decisions(self, tweet_id):
        """
        Retrieves the last two classifications for a given tweet from the taggers_decisions table.

        Parameters:
        - tweet_id (str): The ID of the tweet.

        Returns:
        - A list of dictionaries containing 'classification' and 'features' of the last two classifications.
        """
        tweet_id = str(tweet_id)  # Ensure tweet_id is a string

        with Session(self.engine) as session:
            decisions = (
                session.query(TaggersDecision.classification, TaggersDecision.features)
                .filter(TaggersDecision.tweet_id == tweet_id)
                .order_by(TaggersDecision.tagging_date.desc())  # Get latest decisions first
                .limit(2)  # Only retrieve the last two classifications # TODO check why 2
                .all()
            )

        # Convert to a list of dictionaries
        return [{"classification": d.classification, "features": d.features} for d in decisions]

    # Inserts a tagging result into the table
    def insert_to_tagging_results(self, tweet_id, tag_result, features, decision_source):
        # Ensures tweet_id is a string to match the type in the DB
        tweet_id = str(tweet_id)

        with Session(self.engine) as session:
            # Check if tweet_id already exists in tagging_results
            exists = session.query(TaggingResult).filter(TaggingResult.tweet_id == tweet_id).first()

            if exists:
                return False  # Tweet already classified, no need to insert

            # Insert new record into tagging_results
            new_entry = TaggingResult(
                tweet_id=tweet_id,
                tag_result=tag_result,
                features=features if features else [],  # Ensure it's a list
                decision_source=decision_source
            )

            session.add(new_entry)
            session.commit()
            # Indicates successful insertion
            return True


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





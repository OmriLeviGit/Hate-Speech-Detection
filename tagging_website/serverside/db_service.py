import os
from pathlib import Path

import pytz
from dotenv import load_dotenv
from sqlalchemy import Engine, Nullable, func, text, create_engine, func, or_, and_, exists, select, Boolean, distinct
from sqlalchemy.sql import expression
from sqlalchemy.orm import Session, session

from model import *
from model import User, Tweet, AssignedTweet, TaggersDecision, TaggingResult
from helper_functions.utils import fix_corrupted_text


load_dotenv(os.path.join(Path(__file__).parent.absolute(), '.env.local'))
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
        Base.metadata.create_all(self.engine)


    # Creates a user and generates a password to them
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
            return user.due_date if user else None


    # Returns the user's left_to_classify from the database
    def tweets_left_to_classify(self, user_id):
        with Session(self.engine) as session:
            user = session.query(User).filter(User.user_id == user_id).first()
            
            if not user:
                return None
                
            if user.professional:
                # For professional users, return count of assigned tweets that hasn't been tagged yet
                assigned_count = session.query(func.count(AssignedTweet.tweet_id))\
                    .filter(
                        AssignedTweet.user_id == user_id,
                        AssignedTweet.completed == False
                    ).scalar()
                return assigned_count

            return user.left_to_classify


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

        if not processed_content:   # If could not be parsed correctly
            print(f"could not parse tweet {tweet_id}")
            return False

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

        return True

    # Assigns a specific tweet to the pro user with the least tweets assigned\completed by him
    def assign_tweet_to_pro(self, tweet_id):
        """Assigns a specific tweet to the pro user with the least amount of tweets assigned \\ completed by him"""

        with Session(self.engine) as session:
            pro_user_with_least_work = (
                session.query(
                    User,
                    func.count(distinct(TaggersDecision.tweet_id)).label('tagged_count'),
                    func.count(distinct(AssignedTweet.tweet_id)).label('assigned_count'),
                    (func.count(distinct(TaggersDecision.tweet_id)) + 
                    func.count(distinct(AssignedTweet.tweet_id))).label('total_count')
                )
                .outerjoin(TaggersDecision, User.user_id == TaggersDecision.tagged_by)
                .outerjoin(AssignedTweet, User.user_id == AssignedTweet.user_id)
                .filter(User.professional == True)
                .group_by(User.user_id)
                .order_by('total_count')
                .first()
            )

            user = pro_user_with_least_work[0]

            assignment = AssignedTweet(
                tweet_id=tweet_id,
                user_id=user.user_id,
                completed=False
            )

            session.add(assignment)
            session.commit()


    def get_or_assign_tweet(self, user_id):
        """Get an already assigned tweet or assign a new one to the user and return it."""
        # First check if user already has an assigned tweet
        with Session(self.engine) as session:
            # Join with Tweet to get the full tweet data
            assigned_tweet = (
                session.query(Tweet)
                .join(AssignedTweet, Tweet.tweet_id == AssignedTweet.tweet_id)
                .filter(
                    AssignedTweet.user_id == user_id,
                    AssignedTweet.completed == False
                )
                .first()
            )
            
            if assigned_tweet:
                return {
                    'id': assigned_tweet.tweet_id,
                    'content': assigned_tweet.content,
                    'tweet_url': assigned_tweet.tweet_url
                }

            # Find and assign a new tweet
            new_assigned_tweet = self._find_valid_tweet(session, user_id)

            if not new_assigned_tweet:
                return None  # No available tweet / pro user
   
            # Create the assignment
            assignment = AssignedTweet(
                tweet_id=new_assigned_tweet.tweet_id,
                user_id=user_id,
                completed=False
            )
            session.add(assignment)

            tweet_data = {
                'id': new_assigned_tweet.tweet_id,
                'content': new_assigned_tweet.content,
                'tweet_url': new_assigned_tweet.tweet_url
            }

            session.commit()
            return tweet_data
                

    def _find_valid_tweet(self, session, user_id):
        """Find a valid tweet for assignment for regular users."""

        if self.is_pro(user_id) and self.tweets_left_to_classify(user_id) < 1:
            print("here")
            return None
        print("out", self.tweets_left_to_classify(user_id))

        previously_tagged = session.query(TaggersDecision.tweet_id).filter(
            TaggersDecision.tagged_by == user_id
        ).subquery()

        query = (
            session.query(Tweet)
            .outerjoin(TaggingResult, Tweet.tweet_id == TaggingResult.tweet_id)
            .outerjoin(AssignedTweet, Tweet.tweet_id == AssignedTweet.tweet_id)
            .outerjoin(User, AssignedTweet.user_id == User.user_id)
            .filter(TaggingResult.id.is_(None))
            .filter(~Tweet.tweet_id.in_(previously_tagged.select()))
            .group_by(Tweet.tweet_id)
            .having(func.count(AssignedTweet.tweet_id) <= 1)
            .having(or_(
                func.count(User.professional) == 0,  # No users assigned at all
                ~func.bool_or(User.professional)     # Or no professional users
            ))
            .order_by(func.random())
            .first()
        )

        return query


    # Inserts a tagging decision to the taggers_decisions table
    def insert_to_taggers_decisions(self, tweet_id, user_id, tag_result, features, tagging_duration):
        # Ensures tweet_id is a string to match the type on the DB
        tweet_id = str(tweet_id)
        # Convert seconds (float) to PostgreSQL INTERVAL
        duration_interval = text(f"INTERVAL '{tagging_duration} seconds'")

        with Session(self.engine) as session:
            # Inserts new record into taggers_decisions
            new_entry = TaggersDecision(
                tweet_id=tweet_id,
                tagged_by=user_id,
                classification=tag_result,
                features=features if features else [],  # Ensure it's a list
                tagging_date=datetime.now(pytz.timezone('Asia/Jerusalem')),
                tagging_duration=duration_interval
            )

            session.add(new_entry)
            
            # Remove from assigned_tweets table
            assigned_tweet = session.query(AssignedTweet).filter(
                AssignedTweet.tweet_id == tweet_id,
                AssignedTweet.user_id == user_id
            ).first()

            if assigned_tweet:
                session.delete(assigned_tweet)
            session.commit()


    # Retrieves the last two classifications for a given tweet from the taggers_decisions table
    def get_tweet_decisions(self, tweet_id):
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


    # Returns the number of classifications marked as "Positive" made by a specific user
    def get_positive_classification_count(self, user_id):
        with Session(self.engine) as session:
            return (
                session.query(TaggersDecision)
                .filter(TaggersDecision.tagged_by == user_id)
                .filter(TaggersDecision.classification == "Positive").count())


    # Returns the number of classifications marked as "Negative" made by a specific user from taggers_decisions table
    def get_negative_classification_count(self, user_id):
        with Session(self.engine) as session:
            return (
                session.query(TaggersDecision)
                .filter(TaggersDecision.tagged_by == user_id)
                .filter(TaggersDecision.classification == "Negative")
                .count())


    # Returns the number of classifications marked as "Uncertain" made by a specific user from taggers_decisions table
    def get_uncertain_classification_count(self, user_id):
        with Session(self.engine) as session:
            return (
                session.query(TaggersDecision)
                .filter(TaggersDecision.tagged_by == user_id)
                .filter(TaggersDecision.classification == "Uncertain").count())


    # Returns the number of classifications marked as "Irrelevant" made by a specific user from taggers_decisions table
    def get_irrelevant_classification_count(self, user_id):
        with Session(self.engine) as session:
            return (
                session.query(TaggersDecision)
                .filter(TaggersDecision.tagged_by == user_id)
                .filter(TaggersDecision.classification == "Irrelevant").count())


    # Calculates the average tagging duration for a given user
    def get_average_classification_time(self, user_id):
        with Session(self.engine) as session:
            avg_duration = (
                session.query(func.avg(TaggersDecision.tagging_duration))
                .filter(TaggersDecision.tagged_by == user_id)
                .scalar()
            )

        return avg_duration if avg_duration is not None else 0  # Return 0 if no data


    # Returns the total number of classifications in the taggers_decisions table
    def get_total_classifications(self):
        with Session(self.engine) as session:
            total_tags = (
                session.query(func.count(TaggersDecision.tagger_decision_id))  # Count rows in the table
                .scalar()
            )

        return total_tags if total_tags is not None else 0  # Return 0 if there's no data


    # Returns the total number of classifications marked as "Negative" in taggers_decisions
    def get_total_negative_classifications(self):
        with Session(self.engine) as session:
            total_negative_tags = (
                session.query(func.count(TaggersDecision.tagger_decision_id))  # Count rows
                .filter(TaggersDecision.classification == "Negative")  # Filter by classification
                .scalar()
            )

        return total_negative_tags if total_negative_tags is not None else 0  # Return 0 if no data

    # Returns the total number of classifications marked as "Negative" in taggers_decisions
    def get_total_positive_classifications(self):
        with Session(self.engine) as session:
            total_positive_tags = (
                session.query(func.count(TaggersDecision.tagger_decision_id))  # Count rows
                .filter(TaggersDecision.classification == "Positive")  # Filter by classification
                .scalar()
            )

        return total_positive_tags if total_positive_tags is not None else 0  # Return 0 if no data


    # Returns the total number of classifications marked as "Uncertain" in taggers_decisions
    def get_total_uncertain_classifications(self):
        with Session(self.engine) as session:
            total_positive_tags = (
                session.query(func.count(TaggersDecision.tagger_decision_id))  # Count rows
                .filter(TaggersDecision.classification == "Uncertain")  # Filter by classification
                .scalar()
            )

        return total_positive_tags if total_positive_tags is not None else 0  # Return 0 if no data


    # Returns the total number of classifications marked as "Irrelevant" in taggers_decisions
    def get_total_irrelevant_classifications(self):
        with Session(self.engine) as session:
            total_positive_tags = (
                session.query(func.count(TaggersDecision.tagger_decision_id))  # Count rows
                .filter(TaggersDecision.classification == "Irrelevant")  # Filter by classification
                .scalar()
            )

        return total_positive_tags if total_positive_tags is not None else 0  # Return 0 if no data


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


    # Returns the number of tweets tagged by a specific user
    def count_tags_made(self, user_id):
        with Session(self.engine) as session:
            return session.query(TaggersDecision).filter(TaggersDecision.tagged_by == user_id).count()


    def get_tagged_users(self, min_count=1):
        with Session(self.engine) as session:
            return session.query(
                func.unnest(Tweet.tagged_users).label('tagged_user'), func.count('*').label('count')
            ).group_by('tagged_user')\
            .having(func.count('*') >= min_count)\
            .order_by(func.count('*').desc()).all()


    def get_all_posters(self):
        with Session(self.engine) as session:
            return session.query(
                Tweet.user_posted
            ).order_by(Tweet.user_posted).all()


    def get_hashtags(self, min_count=1):
        with Session(self.engine) as session:
            return session.query(
                func.unnest(Tweet.hashtags).label('hashtag'),
                func.count('*').label('count')
            ).group_by('hashtag')\
            .having(func.count('*') >= min_count)\
            .order_by(func.count('*').desc()).all()
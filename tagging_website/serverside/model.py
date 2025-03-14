from sqlalchemy import (
    Column, Integer, String, Text, Date, Boolean, ForeignKey, TIMESTAMP, ARRAY
)
from sqlalchemy.dialects.postgresql import INTERVAL
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Tweet(Base):
    __tablename__ = 'tweets'
    __table_args__ = {'schema': 'public'}

    tweet_id = Column(Text, primary_key=True)
    user_posted = Column(Text, nullable=True)
    content = Column(Text, nullable=False)
    date_posted = Column(TIMESTAMP, default=datetime.utcnow)
    photos = Column(ARRAY(Text), nullable=True)
    tweet_url = Column(Text, nullable=False)
    tagged_users = Column(ARRAY(Text), nullable=True)
    replies = Column(Integer, default=0)
    reposts = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    views = Column(Integer, default=0)
    hashtags = Column(ARRAY(Text), nullable=True)

    def __repr__(self):
        return f"<Tweet(tweet_id={self.tweet_id}, user_posted={self.user_posted}, content={self.content[:30]}...)>"


class User(Base):
    __tablename__ = 'users'
    __table_args__ = {'schema': 'public'}

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    password = Column(Text, unique=True, nullable=False)
    email = Column(Text, unique=True, nullable=False)
    due_date = Column(Date)
    creation_date = Column(Date, default=datetime.utcnow)
    last_login = Column(TIMESTAMP, nullable=True)
    left_to_classify = Column(Integer, default=0)
    professional = Column(Boolean, default=False)
    current_tweet_id = Column(Text, ForeignKey(Tweet.tweet_id), nullable=True)

    # ToDO - next line might be useful. if not and there's no use of it, delete it
    # current_tweet = relationship("Tweet", back_populates="current_users")

    def __repr__(self):
        return f"<User(user_id={self.user_id}, email={self.email}, professional={self.professional})>"

class TaggersDecision(Base):
    __tablename__ = 'taggers_decisions'
    __table_args__ = {'schema': 'public'}

    tagger_decision_id = Column(Integer, primary_key=True, autoincrement=True)
    tweet_id = Column(Text, ForeignKey(Tweet.tweet_id), nullable=False)
    tagged_by = Column(Integer, ForeignKey(User.user_id), nullable=False)
    classification = Column(Text, nullable=False)
    features = Column(ARRAY(Text), default=[])
    tagging_date = Column(TIMESTAMP, default=datetime.utcnow)
    tagging_duration = Column(INTERVAL)

    def __repr__(self):
        return f"<TaggersDecision(tagger_decision_id={self.tagger_decision_id}, classification={self.classification})>"

class TaggingResult(Base):
    __tablename__ = 'tagging_results'
    __table_args__ = {'schema': 'public'}

    id = Column(Integer, primary_key=True, autoincrement=True)
    tweet_id = Column(Text, ForeignKey(Tweet.tweet_id, ondelete="CASCADE"), nullable=False)
    tag_result = Column(String(255), nullable=False)
    features = Column(ARRAY(Text), default=[])
    decision_source = Column(Text, nullable=True)

    def __repr__(self):
        return f"<TaggersDecision(id={self.id}, tag_result={self.tag_result})>"


class ProBank(Base):
    __tablename__ = 'pro_bank'
    __table_args__ = {'schema': 'public'}

    id = Column(Integer, primary_key=True, autoincrement=True)
    tweet_id = Column(Text, ForeignKey(Tweet.tweet_id), nullable=False)

    def __repr__(self):
        return f"<ProBank(id={self.id}, tweet_id={self.tweet_id}, done={self.done})>"

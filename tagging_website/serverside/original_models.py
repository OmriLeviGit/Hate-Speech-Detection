from datetime import datetime
from sqlalchemy import Column, Integer, String, Date, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    __table_args__ = {'schema': 'public'}

    user_id = Column(Integer, primary_key=True)
    password = Column(Text, unique=True, nullable=False)
    email = Column(Text, unique=True, nullable=False)
    due_date = Column(Date)
    creation_date = Column(Date, default=datetime.utcnow)
    last_login = Column(DateTime(timezone=True), default=datetime.utcnow)
    left_to_classify = Column(Integer, default=0)
    professional = Column(Boolean, default=False)
    current_tweet_tag = Column(Text, ForeignKey('tweets.tweet_id'), nullable=True)

    def __repr__(self):
        return f"<Passcode(" \
               f"id={self.id}, key={self.key}," \
               f" valid_until={self.valid_until}," \
               f" created={self.created}," \
               f" activated={self.activated})," \
               f" email={self.email})," \
               f" max_classifications={self.max_classifications})>"
    
    
    def is_valid(self, num_classifications):
        current_date = datetime.now().date()
        return (self.valid_until > current_date and
                self.activated and
                (self.max_classifications is None or num_classifications < self.max_classifications))


class Tweeter(Base):
    __tablename__ = 'tweeters'
    __table_args__ = {'schema': 'public'}

    username = Column(String, primary_key=True)
    

    def __repr__(self):
        return f"<Tweeter(username={self.username})>"


class Tweet(Base):
    __tablename__ = 'tweets'
    __table_args__ = {'schema': 'public'}

    id = Column(String, primary_key=True)
    tweeter = Column(ForeignKey(Tweeter.username))
    content = Column(String)

    def __repr__(self):
        return f"<Tweet(id={self.id})>"


class Classification(Base):
    __tablename__ = 'classifications'
    __table_args__ = {'schema': 'public'}

    id = Column(Integer, primary_key=True, autoincrement=True)
    tweet = Column(ForeignKey(Tweet.id))
    classifier = Column(ForeignKey(Passcode.key))
    classification = Column(String)
    features = Column(String, default="")
    classified_at = Column(Date, default=datetime.now())
    started_classification = Column(Date, default=datetime.now())

    def __repr__(self):
        return f"<Classification(id={self.id}, tweet={self.tweet}, classifier={self.classifier}, classification={self.classification})>"


class ProBank(Base):
    __tablename__ = 'pro_bank'
    __table_args__ = {'schema': 'public'}

    id = Column(Integer, primary_key=True, autoincrement=True)
    tweet = Column(ForeignKey(Tweet.id))
    done = Column(Boolean, default=False)

    def __repr__(self):
        return f"<ProBank(id={self.id}, tweet={self.tweet}, done={self.done})>"


-- Connect to tagger_db first, then run this to drop all tables
DO $$ 
DECLARE
    r RECORD;
BEGIN
    -- Disable foreign key checks during table deletion
    EXECUTE 'SET CONSTRAINTS ALL DEFERRED';
    
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
        EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;
    
    -- Re-enable foreign key checks
    EXECUTE 'SET CONSTRAINTS ALL IMMEDIATE';
END $$;

CREATE TABLE tweets (
    tweet_id TEXT PRIMARY KEY,
    user_posted TEXT,
    content TEXT,
    date_posted TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    photos TEXT[],
    tweet_url TEXT,
    tagged_users TEXT[],
    replies INT DEFAULT 0,
    reposts INT DEFAULT 0,
    likes INT DEFAULT 0,
    views INT DEFAULT 0,
    hashtags TEXT[]
);

CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    password TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    due_date DATE,
    creation_date DATE DEFAULT CURRENT_DATE,
    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    left_to_classify INT DEFAULT 0,
    professional BOOLEAN DEFAULT FALSE,
    current_tweet_id TEXT DEFAULT NULL REFERENCES tweets(tweet_id)
);

CREATE TABLE taggers_decisions (
    tagger_decision_id SERIAL PRIMARY KEY,
    tweet_id TEXT REFERENCES tweets(tweet_id) ON DELETE CASCADE,
    tagged_by INT REFERENCES users(user_id) ON DELETE SET NULL,
    classification TEXT NOT NULL,
    features TEXT[],
    tagging_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tagging_duration INTERVAL
);

CREATE TABLE pro_bank (
    id SERIAL PRIMARY KEY,
    tweet_id TEXT REFERENCES tweets(tweet_id) ON DELETE CASCADE
);

CREATE TABLE tagging_results (
    id SERIAL PRIMARY KEY,
    tweet_id TEXT REFERENCES tweets(tweet_id) ON DELETE CASCADE,
    tag_result VARCHAR(255) NOT NULL,
    features TEXT,
    decision_source TEXT
);

-- Indexes for faster queries
CREATE INDEX idx_tweets_user ON tweets(user_posted);
CREATE INDEX idx_annotations_tweet ON taggers_decisions(tweet_id);
CREATE INDEX idx_annotations_user ON taggers_decisions(tagged_by);
CREATE INDEX idx_pro_bank_tweet ON pro_bank(tweet_id);
CREATE INDEX idx_results_tweet ON tagging_results(tweet_id);

-- Confirm creation
SELECT 'Database and tables created successfully!' AS confirmation;




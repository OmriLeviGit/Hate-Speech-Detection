import psycopg2
from generate_users import generate_users
from import_tweets import import_tweets_from_csv

def reset_database():
    db_params = {
        'dbname': 'tagger_db',
        'user': 'postgres',
        'password': '1234',
        'host': 'localhost',
        'port': '5432'
    }

    try:
        conn = psycopg2.connect(**db_params)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # drop all tables
        drop_tables_sql = """
        DO $$
        DECLARE
            r RECORD;
        BEGIN
            --* Disable foreign key checks during table deletion*
            EXECUTE 'SET CONSTRAINTS ALL DEFERRED';
           
            FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
                EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
            END LOOP;
           
            --* Re-enable foreign key checks*
            EXECUTE 'SET CONSTRAINTS ALL IMMEDIATE';
        END $$;
        """
        cursor.execute(drop_tables_sql)
        
        # Create tweets table
        cursor.execute("""
        CREATE TABLE public.tweets (
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
        """)
        
        # Create users table
        cursor.execute("""
        CREATE TABLE public.users (
            user_id SERIAL PRIMARY KEY,
            password TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            due_date DATE DEFAULT NULL,
            creation_date DATE DEFAULT CURRENT_DATE,
            last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            left_to_classify INT DEFAULT 0,
            professional BOOLEAN DEFAULT FALSE
        );
        """)
        
        # Create assigned_tweets table
        cursor.execute("""
        CREATE TABLE public.assigned_tweets (
            user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
            tweet_id TEXT REFERENCES tweets(tweet_id) ON DELETE CASCADE,
            completed BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (user_id, tweet_id)
        );
        """)
        
        # Create taggers_decisions table
        cursor.execute("""
        CREATE TABLE public.taggers_decisions (
            tagger_decision_id SERIAL PRIMARY KEY,
            tweet_id TEXT REFERENCES tweets(tweet_id) ON DELETE CASCADE,
            tagged_by INT REFERENCES users(user_id) ON DELETE SET NULL,
            classification TEXT NOT NULL,
            features TEXT[],
            tagging_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tagging_duration INTERVAL
        );
        """)
        
        # Create tagging_results table
        cursor.execute("""
        CREATE TABLE public.tagging_results (
            id SERIAL PRIMARY KEY,
            tweet_id TEXT REFERENCES tweets(tweet_id) ON DELETE CASCADE,
            tag_result VARCHAR(255) NOT NULL,
            features TEXT,
            decision_source TEXT
        );
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX idx_tweets_user ON tweets(user_posted);")
        cursor.execute("CREATE INDEX idx_annotations_tweet ON taggers_decisions(tweet_id);")
        cursor.execute("CREATE INDEX idx_annotations_user ON taggers_decisions(tagged_by);")
        cursor.execute("CREATE INDEX idx_results_tweet ON tagging_results(tweet_id);")
        
        # Confirm creation
        cursor.execute("SELECT 'Database and tables created successfully!' AS confirmation;")
        result = cursor.fetchone()
        print(result[0])
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the cursor and connection
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    reset_database()
    # generate_users()
    # import_tweets_from_csv()

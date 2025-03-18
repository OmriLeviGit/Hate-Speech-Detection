from reset_database import reset_database
from import_and_generate_taggers import generate_users
from import_tweets import import_tweets_from_csv


def initialize(batch_1=None):
    reset_database()
    tweets_per_user = 125
    num_of_users = 14
    generate_users(5, tweets_per_user)
    total_tweets = tweets_per_user * num_of_users // 2
    import_tweets_from_csv(file_name=batch_1, limit=total_tweets)
    print("DB was reset and loaded")

if __name__ == "__main__":
    # sudo service postgresql start
    batch_1 = "antisemistic_batch1_bd_20250315_175811_0.csv"
    initialize(batch_1)
    
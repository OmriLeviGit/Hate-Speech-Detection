from reset_database import reset_database
from import_and_generate_taggers import generate_users
from import_tweets import import_tweets_from_csv


def initialize(batch_1=None):
    reset_database()
    generate_users()
    import_tweets_from_csv(batch_1)
    print("DB was reset and loaded")

if __name__ == "__main__":
    # sudo service postgresql start
    batch_1 = "antisemistic_batch1_bd_20250315_175811_0.csv"

    initialize(batch_1)
    
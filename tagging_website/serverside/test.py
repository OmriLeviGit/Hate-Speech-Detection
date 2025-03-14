from model import *
# from serverside import db_service
from controller import *

# db = db_service.get_instance()


# Test insert tweet
# date_posted = datetime.today()
# content = "ðŸ“Grampians, Victoria. @NSNContactBot @WhiteAusVic"
# db.insert_tweet(
#     "111", "Tomer",
#     content, date_posted,
#     ["photo1.jpg", "photo2.jpg"],
#     "http://twitter.com/test_tweet",
#     ["user1", "user2"],
#     5,3,100,500, ["#Nero", "#Bero"],
# )

# Test get_Tweet
# tweet = db.get_tweet(1)

# Test create_user
# user = db.create_user("test5@gmail.com", 12, 20)

# Test is_pro
# is_pro = db.is_pro(1)
# print(is_pro)

# Test insert_to_pro_bank
# db.insert_tweet_to_pro_bank("123")

# Test insert_to_tagging_results
# success = db.insert_to_tagging_results(
#     tweet_id="123",
#     tag_result="Positive",
#     features=["hate_speech", "violence"],
#     decision_source="Expert Review"
# )
#
# if success:
#     print("Tagging result added successfully!")
# else:
#     print("Tweet already exists in tagging_results.")

# Test


import asyncio
from controller import handle_tweet_tagging
from asyncio.locks import Lock

async def main():
    lock = Lock()

    # Part 1 - Make a tweet to go to pro_bank by conflicting two regular users decisions
    # Result1
    user_id1 = 1
    tweet_id1 = "a1"
    classification1 = "Positive"
    features1 = ["a", "b"]

    # # Result2
    # user_id2 = 2
    # tweet_id2 = "a2"
    # classification2 = "Uncertain"
    # features2 = ["a", "c"]
    #
    # # Run the function
    await handle_tweet_tagging(lock, user_id1, tweet_id1, classification1, features1)
    # await handle_tweet_tagging(lock, user_id2, tweet_id2, classification2, features2)


    # Part 2 - Check Pro user tagging abilities
    # Result3 - Pro User
    # user_id = 3
    # tweet_id = "a2"
    # classification = "Positive"
    # features = ["a", "b"]
    # await handle_tweet_tagging(lock, user_id, tweet_id, classification, features)

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())





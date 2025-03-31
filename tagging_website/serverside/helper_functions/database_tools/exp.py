import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:1234@localhost:5432/tagger_db")
table_name = "taggers_decisions"
table_name = "tagging_results"
query = f"""
SELECT 
    td.*, 
    t.content
FROM 
    taggers_decisions td
JOIN 
    tweets t ON td.tweet_id = t.tweet_id
WHERE 
    td.tagged_by = 6
"""
df = pd.read_sql(query, engine)
df.to_csv("tomer_tags.csv", index=False)



# The next query returns all the tagging results and the content of each tweet, filetered by a specific user and in a defined time frame
# 
# SELECT 
#     td.*, 
#     t.content
# FROM 
#     taggers_decisions td
# JOIN 
#     tweets t ON td.tweet_id = t.tweet_id
# WHERE 
#     td.tagged_by = 6
#     AND td.tagging_date >= TIMESTAMP '2025-03-26 00:00:00'
#     AND td.tagging_date < TIMESTAMP '2025-03-27 00:00:00'


# The next query returns all the tagging results and the content of each tweet
#
# SELECT 
#     tr.*, 
#     t.content
# FROM 
#     tagging_results tr
# JOIN 
#     tweets t ON tr.tweet_id = t.tweet_id
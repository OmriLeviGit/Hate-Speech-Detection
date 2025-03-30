import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:1234@localhost:5432/tagger_db")
table_name = "taggers_decisions"
table_name = "tagging_results"
query = f"""
SELECT * FROM tweets
"""
df = pd.read_sql(query, engine)
df.to_csv("tweets_27_03_2025.csv", index=False)



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
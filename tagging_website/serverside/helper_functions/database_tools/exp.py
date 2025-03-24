import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:1234@localhost:5432/tagger_db")
table_name = "taggers_decisions"
table_name = "tagging_results"
query = f"""
select distinct tweet_id
from taggers_decisions;
"""
df = pd.read_sql(query, engine)
df.to_csv("count.csv", index=False)

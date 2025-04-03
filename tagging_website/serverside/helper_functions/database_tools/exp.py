import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:1234@localhost:5432/tagger_db")
table_name = "taggers_decisions"
table_name = "tagging_results"
query = f"""
SELECT
  u.email,
  COUNT(*) AS total_classifications,
  ROUND(AVG(EXTRACT(EPOCH FROM td.tagging_duration))) AS avg_duration_seconds,
  COUNT(*) FILTER (WHERE td.classification = 'Positive') AS num_positive,
  COUNT(*) FILTER (WHERE td.classification = 'Negative') AS num_negative,
  COUNT(*) FILTER (WHERE td.classification = 'Irrelevant') AS num_irrelevant,
  COUNT(*) FILTER (WHERE td.classification = 'Uncertain') AS num_uncertain,
  ROUND(1.0 * COUNT(*) FILTER (WHERE td.classification = 'Positive') / COUNT(*), 4) AS percent_positive,
  ROUND(1.0 * COUNT(*) FILTER (WHERE td.classification = 'Negative') / COUNT(*), 4) AS percent_negative,
  ROUND(1.0 * COUNT(*) FILTER (WHERE td.classification = 'Irrelevant') / COUNT(*), 4) AS percent_irrelevant,
  ROUND(1.0 * COUNT(*) FILTER (WHERE td.classification = 'Uncertain') / COUNT(*), 4) AS percent_uncertain
FROM taggers_decisions td
JOIN users u ON td.tagged_by = u.user_id
WHERE td.tagging_date >= TIMESTAMP '2025-03-27 00:00:00'
GROUP BY u.email
ORDER BY u.email;

"""
df = pd.read_sql(query, engine)
df.to_csv("second_round_until_april_1st.csv", index=False)





# Get all tags made by a specific user
# SELECT 
#     td.*, 
#     t.content
# FROM 
#     taggers_decisions td
# JOIN 
#     tweets t ON td.tweet_id = t.tweet_id
# WHERE 
#     td.tagged_by = 6


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
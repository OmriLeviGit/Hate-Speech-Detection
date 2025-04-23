
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from db_service import get_db_instance

def main():
    db = get_db_instance()
    positive_texts = db.get_result_posts("Positive")
    negative_texts = db.get_result_posts("Negative")
    irrelevant_texts = db.get_result_posts("Irrelevant")

    pd.DataFrame(negative_texts).to_csv("negative_results.csv", index=False, header=False)
    pd.DataFrame(positive_texts).to_csv("positive_results.csv", index=False, header=False)
    pd.DataFrame(irrelevant_texts).to_csv("irrelevant_results.csv", index=False, header=False)

if __name__ == "__main__":
    main()

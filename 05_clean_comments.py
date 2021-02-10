import pandas as pd
import re
from html import unescape
from joblib import Parallel, delayed

YEAR = 2019
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
CMT_DF_BASE_DIR = "data/processed/comments/"
MAX_PROCESSES = 12

def clean_text(dirty_text):
    clean_text = "".join(x for x in dirty_text if x.isprintable())
    cleaned_text = clean_text.replace("&gt;", "")
    cleaned_text = cleaned_text.replace("&amp;", "")
    cleaned_text = re.sub("\s\s+", " ", cleaned_text).replace("\n", " ").strip()
    cleaned_text = re.sub(r"(http|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", "", cleaned_text)
    cleaned_text = unescape(cleaned_text).lower()
    return cleaned_text

def process_comments(month):
    print("month {}: loading comments".format(month))
    comments_df = pd.read_pickle("{}filtered-{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))
    print("month {}: cleaning comments".format(month))
    comments_df["text"] = comments_df["text"].apply(clean_text)
    print("month {}: saving comments".format(month))
    comments_df.to_pickle("{}cleaned-{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))

if __name__ == "__main__":
    print("starting parallel execution")
    Parallel(n_jobs=MAX_PROCESSES)(delayed(process_comments)(month) for month in MONTHS)

    print("done")

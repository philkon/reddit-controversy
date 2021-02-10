import pandas as pd
import zstandard
import io
import json
import itertools
from joblib import Parallel, delayed

EXTRACT_SUBMISSIONS = True
EXTRACT_COMMENTS = True
SUB_SOURCE_DIR = "data/raw/submissions/"
SUB_DF_BASE_DIR = "data/processed/submissions/"
CMT_SOURCE_DIR = "data/raw/comments/"
CMT_DF_BASE_DIR = "data/processed/comments/"
YEAR = 2019
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
MAX_PROCESSES = 3
SUBREDDITS_PER_LANGUAGE = {
    "de": ["de", "Austria", "ich_iel", "FragReddit", "wasletztepreis", "de_IAmA", "Dachschaden", "Finanzen", "rocketbeans"],
    "en": ["AskReddit", "politics", "nfl", "worldnews", "funny", "me_irl", "todayilearned", "technology", "philosophy", "StarWars", "sports", "facepalm"],
    "es": ["es", "espanol", "argentina", "mexico", "chile", "spain", "vzla", "podemos", "uruguay", "Colombia", "yo_elvr"],
    "fr": ["france", "rance", "Quebec", "montreal", "jeuxvideo", "FranceLibre"],
    "it": ["italy", "ItalyInformatica", "litigi", "Italia", "Libri"],
    "pt": ["portugal", "PORTUGALCARALHO", "brasil", "BrasildoB", "circojeca", "brasilivre", "PrimeiraLiga"]
}
SUBREDDITS_LIST = list(itertools.chain.from_iterable(SUBREDDITS_PER_LANGUAGE.values()))
SUBREDDIT_LANGUAGE_MAPPING = {}
for language, subreddit_list in SUBREDDITS_PER_LANGUAGE.items():
    for subreddit in subreddit_list:
        SUBREDDIT_LANGUAGE_MAPPING[subreddit] = language

def extract_submissions(month):
    month_results = []
    sub_file = "{}RS_{}-{:02d}.zst".format(SUB_SOURCE_DIR, YEAR, month)
    
    with open(sub_file, 'rb') as fh:
        dctx = zstandard.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        for line in text_stream:
            json_line = json.loads(line)
            try:
                subreddit = json_line["subreddit"]
            except KeyError:
                continue
            if subreddit not in SUBREDDITS_LIST:
                continue
            line_result = {
                "subreddit": subreddit,
                "subreddit_language": SUBREDDIT_LANGUAGE_MAPPING[subreddit],
                "title": json_line["title"],
                "text": json_line["selftext"],
                "type": "submission",
                "author": json_line["author"],
                "timestamp": json_line["created_utc"],
                "id": json_line["id"],
                "score": json_line["score"],
                "is_self": json_line["is_self"],
                "is_video": json_line["is_video"],
                "is_original_content": json_line["is_original_content"]
            }
            month_results.append(line_result)
    month_results_df = pd.DataFrame(month_results)
    month_results_df.to_pickle("{}{}-{}.p".format(SUB_DF_BASE_DIR, YEAR, month))

def extract_comments(month):
    month_results = []
    cmt_file = "{}RC_{}-{:02d}.zst".format(CMT_SOURCE_DIR, YEAR, month)
    
    with open(cmt_file, 'rb') as fh:
        dctx = zstandard.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        for line in text_stream:
            json_line = json.loads(line)
            try:
                subreddit = json_line["subreddit"]
            except KeyError:
                continue
            if subreddit not in SUBREDDITS_LIST:
                continue
            line_result = {
                "subreddit": subreddit,
                "subreddit_language": SUBREDDIT_LANGUAGE_MAPPING[subreddit],
                "text": json_line["body"],
                "type": "comment",
                "author": json_line["author"],
                "timestamp": json_line["created_utc"],
                "id": json_line["id"],
                "parent_id": json_line["parent_id"][3:],
                "controversial": json_line["controversiality"],
                "score": json_line["score"]
            }
            month_results.append(line_result)
    month_results_df = pd.DataFrame(month_results)
    month_results_df.to_pickle("{}{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))

if __name__ == "__main__":
    if EXTRACT_SUBMISSIONS:
        print("parsing submissions...")
        Parallel(n_jobs=MAX_PROCESSES)(delayed(extract_submissions)(month) for month in MONTHS)
        print("done")
    
    if EXTRACT_COMMENTS:
        print("parsing comments...")
        Parallel(n_jobs=MAX_PROCESSES)(delayed(extract_comments)(month) for month in MONTHS)
        print("done")

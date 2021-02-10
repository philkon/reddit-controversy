import pandas as pd
import igraph as ig
from tqdm import tqdm

YEAR = 2019
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
SUB_DF_BASE_DIR = "data/processed/submissions/"
CMT_DF_BASE_DIR = "data/processed/comments/"
GRAPH_FILENAME = "data/processed/reddit_submissions_graph.p"
MIN_NUM_COMMENTS = 10
FILTERED_SUBMISSIONS_FILENAME = "data/processed/submissions/filtered_reddit_submissions.p"

if __name__ == "__main__":
    print("loading graph")
    g = ig.Graph.Read_Pickle(GRAPH_FILENAME)
    
    print("parsing submissions")
    submission_comment_ids = {}
    submission_details = []
    for month in MONTHS:
        submission_df = pd.read_pickle("{}{}-{}.p".format(SUB_DF_BASE_DIR, YEAR, month))
        id_index = list(submission_df.columns).index("id") + 1
        subreddit_index = list(submission_df.columns).index("subreddit") + 1
        subreddit_language_index = list(submission_df.columns).index("subreddit_language") + 1
        author_index = list(submission_df.columns).index("author") + 1
        timestamp_index = list(submission_df.columns).index("timestamp") + 1
        title_index = list(submission_df.columns).index("title") + 1
        for row in tqdm(submission_df.itertuples(), total=len(submission_df), desc="month {}".format(month)):
            successors = g.neighborhood(row[id_index], mode="out", order=100000000)[1:]
            num_successors = len(successors)
            if num_successors < MIN_NUM_COMMENTS:
                continue
            successors_vs = ig.VertexSeq(g, successors)
            submission_comment_ids.update(dict.fromkeys(successors_vs["name"], row[id_index]))
            submission_details.append({"id": row[id_index], "subreddit": row[subreddit_index], "subreddit_language": row[subreddit_language_index], "author": row[author_index], "timestamp": row[timestamp_index], "title": row[title_index]})
    submissions_df = pd.DataFrame(submission_details)
    submissions_df.set_index("id", inplace=True)
    print("saving filtered submissions")
    submissions_df.to_pickle(FILTERED_SUBMISSIONS_FILENAME)
     
    print("parsing comments")
    comments_df = pd.DataFrame()
    for month in MONTHS:   
        print("loading month {}".format(month))
        month_comments_df = pd.read_pickle("{}{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))
        month_comments_df.set_index("id", inplace=True)
        month_comments_df["submission_id"] = pd.Series(submission_comment_ids)
        print("num comments before filtering without submission: {}".format(len(month_comments_df)))
        month_comments_df.dropna(subset=["submission_id"], inplace=True)
        print("num comments after filtering without submission: {}".format(len(month_comments_df)))
        print("saving filtered month {}".format(month))
        month_comments_df.to_pickle("{}filtered-{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))

    print("done")

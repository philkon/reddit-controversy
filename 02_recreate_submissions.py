import pandas as pd
import igraph as ig
from tqdm import tqdm

YEAR = 2019
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
SUB_DF_BASE_DIR = "data/processed/submissions/"
CMT_DF_BASE_DIR = "data/processed/comments/"
GRAPH_FILENAME = "data/processed/reddit_submissions_graph.p"

if __name__ == "__main__":
    g = ig.Graph(directed=True)
    print("adding submission nodes")
    for month in MONTHS:
        print("\tmonth {}".format(month))
        submission_df = pd.read_pickle("{}{}-{}.p".format(SUB_DF_BASE_DIR, YEAR, month))
        g.add_vertices(submission_df["id"])
    print("adding comment nodes")
    for month in MONTHS:   
        print("\tmonth {}".format(month))
        comment_df = pd.read_pickle("{}{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))
        g.add_vertices(comment_df["id"])
    existing_ids = set(g.vs["name"])
    print("number of existing ids:", len(existing_ids))
    print("adding edges")
    for month in MONTHS:
        comment_df = pd.read_pickle("{}{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))
        id_index = list(comment_df.columns).index("id") + 1
        parent_id_index = list(comment_df.columns).index("parent_id") + 1
        edge_list = []
        for row in tqdm(comment_df.itertuples(), total=len(comment_df), desc="month {}".format(month)):
            if row[parent_id_index] in existing_ids:
                edge_list.append((row[parent_id_index], row[id_index]))
        g.add_edges(edge_list)
    print("saving graph")
    g.write_pickle(GRAPH_FILENAME)

    print("done")

import pandas as pd
import igraph as ig
from tqdm import tqdm

YEAR = 2019
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
GRAPH_FILENAME = "data/processed/reddit_submissions_graph.p"
FILTERED_GRAPH_FILENAME = "data/processed/reddit_filtered_graph.p"
FILTERED_SUBMISSIONS_FILENAME = "data/processed/submissions/filtered_reddit_submissions.p"
CMT_DF_BASE_DIR = "data/processed/comments/"

if __name__ == "__main__":
    print("loading graph")
    g = ig.Graph.Read_Pickle(GRAPH_FILENAME)

    index_list = []
    print("loading submission")
    submissions_df = pd.read_pickle(FILTERED_SUBMISSIONS_FILENAME)
    index_list.append(submissions_df.index.to_series())

    print("loading comments")
    for month in tqdm(MONTHS):
        comments_df = pd.read_pickle("{}filtered-{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))
        index_list.append(comments_df.index.to_series())

    print("coputing ids to delete")
    index_s = pd.concat(index_list)
    graph_ids = set(g.vs["name"])
    filtered_ids = set(index_s)
    delete_ids = graph_ids - filtered_ids

    print("deleting vertices ({} to delete)".format(len(delete_ids)))
    print("num before: {}".format(len(g.vs)))
    g.delete_vertices(delete_ids)
    print("num before: {}".format(len(g.vs)))

    print("saving filtered graph")
    g.write_pickle(FILTERED_GRAPH_FILENAME)

    print("done")

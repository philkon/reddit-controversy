import os

if __name__ == "__main__":
    dirs = [
        "data/raw/comments/",
        "data/raw/submissions/",
        "data/processed/comments/",
        "data/processed/submissions/",
        "data/processed/features/",
        "results/plots/"
    ]
    for dir in dirs:
        try:
            os.makedirs(dir)
            print("created: '{}'".format(dir))
        except FileExistsError:
            print("already exists: '{}'".format(dir))

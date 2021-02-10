import pandas as pd
import spacy
from joblib import Parallel, delayed
from spacy_cld import LanguageDetector

YEAR = 2019
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
CMT_DF_BASE_DIR = "data/processed/comments/"
FEATURE_BASE_DIR = "data/processed/features/"
MAX_PROCESSES = 22
DETECT_LANGUAGE = False
DEFINE_LANGUAGE = True
LANGUAGES_TO_USE = [ "de", "en", "es", "fr", "it", "pt"]

def process_comments(month):
    print("month {}: loading comments".format(month))
    comments_df = pd.read_pickle("{}cleaned-{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))
    print("month {}: cleaning comments".format(month))
    comments_df["text"] = comments_df["text"].apply(clean_text)
    print("month {}: saving comments".format(month))
    comments_df.to_pickle("{}cleaned-language-{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))

def language_pipe(doc):
    return {"detected_languages": doc._.languages, "detected_languages_scores": doc._.language_scores}

def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def process_chunk(texts):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=100):
        preproc_pipe.append(language_pipe(doc))
    return preproc_pipe

def preprocess_parallel(texts, chunksize=10000):
    executor = Parallel(n_jobs=MAX_PROCESSES, backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk)
    tasks = (do(chunk) for chunk in chunker(texts, len(comments_df), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)

def filter_languages(row):
    if len(row["detected_languages"]) != 1:
        return None
    language = row["detected_languages"][0]
    if row["detected_languages_scores"][language] < 0.9 or language not in LANGUAGES_TO_USE:
        return None
    else:
        return language

if __name__ == "__main__":
    if DETECT_LANGUAGE:
        print("starting language detection")
        nlp = spacy.blank("en")
        language_detector = LanguageDetector()
        nlp.add_pipe(language_detector)

        for month in MONTHS:
            print("month {}: loading comments".format(month))
            comments_df = pd.read_pickle("{}cleaned-{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))
            print("month {}: processing comments".format(month))
            languages_raw = preprocess_parallel(comments_df["text"], chunksize=1000000)
            languages_df = pd.DataFrame(languages_raw, index=comments_df.index)
            print("month {}: saving languages".format(month))
            languages_df.to_pickle("{}languages-{}-{}.p".format(FEATURE_BASE_DIR, YEAR, month))

        print("done")
    
    if DEFINE_LANGUAGE:
        for month in MONTHS:
            print("month {}: loading languages".format(month))
            languages_df = pd.read_pickle("{}languages-{}-{}.p".format(FEATURE_BASE_DIR, YEAR, month))

            print("month {}: loading comments".format(month))
            comments_df = pd.read_pickle("{}cleaned-{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))

            print("month {}: defining language".format(month))
            comments_df["detected_language"] = languages_df.apply(filter_languages, axis=1)

            print("month {}: saving comments".format(month))
            comments_df.to_pickle("{}final-{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))

        print("done")

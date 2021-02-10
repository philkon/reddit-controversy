import pandas as pd
import numpy as np
import igraph as ig
import spacy
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
from spacy_syllables import SpacySyllables
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import chi2_contingency
from sklearn.feature_extraction import text

YEAR = 2019
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
LANGUAGES = ["de", "en", "es", "fr", "it", "pt"]
CMT_DF_BASE_DIR = "data/processed/comments/"
FEATURE_BASE_DIR = "data/processed/features/"
SENTIMENT_DIR = "data/sentiment/"
FILTERED_GRAPH_FILENAME = "data/processed/reddit_filtered_graph.p"
FILTERED_SUBMISSIONS_FILENAME = "data/processed/submissions/filtered_reddit_submissions.p"
MAX_PROCESSES = 22
COMPUTE_LANGUAGE_FEATURES = True
COMPUTE_STRUCTURAL_FEATURES = True
COMPUTE_TOP_WORDS = True
COMPUTE_CONTENT_FEATURES = True

sentiment_lexica = {}
for lang in LANGUAGES:
    sentiment_lexica[lang] = {}
    with open("{}negative_words_{}.txt".format(SENTIMENT_DIR, lang.lower()), "r") as fr:
        sentiment_lexica[lang]["neg"] = fr.read().lower().splitlines()
    with open("{}positive_words_{}.txt".format(SENTIMENT_DIR, lang.lower()), "r") as fr:
        sentiment_lexica[lang]["pos"] = fr.read().lower().splitlines()

# Basic text stats
def get_basic_stats(doc, lang):
    num_chars = len(doc.text)
    num_syllables = sum([token._.syllables_count for token in doc if token.is_alpha])
    num_words = len([token for token in doc if token.is_alpha])
    num_sentences = len(list(doc.sents))
    characters_to_sentences_ratio = num_chars / num_sentences
    words_to_sentences_ratio = num_words / num_sentences

    if num_words < 100:
        flesch_reading_ease = None
    else:
        syllables_per_100_words = num_syllables * (100 / num_words)
        if lang == "en":
            flesch_reading_ease = 206.835 - (1.015 * words_to_sentences_ratio) - (84.6 * (num_syllables / num_words))
        elif lang == "fr":
            flesch_reading_ease = 207 - (1.015 * words_to_sentences_ratio) - (73.6 * (num_syllables / num_words))
        elif lang == "de":
            flesch_reading_ease = 180 - words_to_sentences_ratio - (58.5 * (num_syllables / num_words))
        elif lang == "it":
            flesch_reading_ease = 217 - (1.3 * words_to_sentences_ratio) - (0.6 * syllables_per_100_words)
        elif lang == "pt":
            flesch_reading_ease = 248.835 - (84.6 * (num_syllables / num_words)) - (1.015 * words_to_sentences_ratio)
        elif lang == "es": 
            flesch_reading_ease = 206.84 - (0.6 * syllables_per_100_words) - (1.02 * words_to_sentences_ratio)
        if flesch_reading_ease > 100:
            flesch_reading_ease = 100
        if flesch_reading_ease < 0:
            flesch_reading_ease = 0

    return {
        "num_chars": num_chars,
        "num_syllables": num_syllables,
        "num_words": num_words,
        "num_sentences": num_sentences,
        "characters_to_sentences_ratio": characters_to_sentences_ratio,
        "words_to_sentences_ratio": words_to_sentences_ratio,
        "flesch_reading_ease": flesch_reading_ease
    }

# POS tags
def get_pos_tags(doc):
    if len(doc) == 0:
        return {
            "pos_ratio_nouns": None,
            "pos_ratio_verbs": None,
            "pos_ratio_adjectives": None,
            "pos_ratio_adverbs": None,
            "pos_ratio_pronouns": None 
        }
    num_nouns = 0
    num_verbs = 0
    num_adjectives = 0
    num_adverbs = 0
    num_pronouns = 0
    for token in doc:
        if token.pos_ == "NOUN": num_nouns += 1
        elif token.pos_ == "VERB": num_verbs += 1
        elif token.pos_ == "ADJ": num_adjectives += 1
        elif token.pos_ == "ADV": num_adverbs += 1
        elif token.pos_ == "PRON": num_pronouns += 1
    return {
        "pos_ratio_nouns": num_nouns / len(doc),
        "pos_ratio_verbs": num_verbs / len(doc),
        "pos_ratio_adjectives": num_adjectives / len(doc),
        "pos_ratio_adverbs": num_adverbs / len(doc),
        "pos_ratio_pronouns": num_pronouns / len(doc) 
    }

# Sentiment
def get_sentiment(doc, sent_dict):
    num_negative = 0
    num_positive = 0
    for token in doc:
        if token.text in sent_dict["neg"]: num_negative += 1
        elif token.text in sent_dict["pos"]: num_positive += 1
    try:
        score = (num_positive - num_negative) / (num_positive + num_negative)
    except ZeroDivisionError:
        score = None
    return {"sentiment": score}

def process_text(text, nlp, sent_dict, lang):
    doc = nlp(text.lower())
    doc_results = get_basic_stats(doc, lang)
    doc_results.update(get_pos_tags(doc))
    doc_results.update(get_sentiment(doc, sent_dict))
    return pd.Series(doc_results)

def process_split(split_df, lang):
    if lang == "de":
        nlp = spacy.load("de_core_news_sm", disable=["ner"])
    elif lang == "en":
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    elif lang == "es":
        nlp = spacy.load("es_core_news_sm", disable=["ner"])
    elif lang == "fr":
        nlp = spacy.load("fr_core_news_sm", disable=["ner"])
    elif lang == "it":
        nlp = spacy.load("it_core_news_sm", disable=["ner"])
    elif lang == "pt":
        nlp = spacy.load("pt_core_news_sm", disable=["ner"])
    syllables = SpacySyllables(nlp)
    nlp.add_pipe(syllables, after="tagger")
    #nlp.add_pipe(Readability())
    sent_dict = sentiment_lexica[lang]
    return split_df["text"].apply(process_text, args=[nlp, sent_dict, lang])

if __name__ == "__main__":
    if COMPUTE_LANGUAGE_FEATURES:
        print("computing language features")
        for month in MONTHS:
            print("month {}: loading comments".format(month))
            comments_df = pd.read_pickle("{}final-{}-{}.p".format(CMT_DF_BASE_DIR, YEAR, month))
            month_features_list = []
            for language in LANGUAGES:
                print("month {}: processing '{}' comments".format(month, language))
                language_df = comments_df[comments_df["detected_language"] == language]
                features_raw = Parallel(n_jobs=MAX_PROCESSES)(delayed(process_split)(df, language) for g, df in language_df.groupby(np.arange(len(language_df)) // (len(language_df) / MAX_PROCESSES)))
                features_df = pd.concat(features_raw)
                month_features_list.append(features_df)
            print("month {}: merging languages".format(month))
            month_features_df = pd.concat(month_features_list)
            print("month {}: merging features with existing data frame".format(month))
            result_df = pd.concat([comments_df, month_features_df], axis=1)
            print("month {}: saving language features".format(month))
            result_df.to_pickle("{}language-features-{}-{}.p".format(FEATURE_BASE_DIR, YEAR, month))
        print("done")

    if COMPUTE_STRUCTURAL_FEATURES:
        print("computing structural features")
        author_list = []
        timestamp_list = []
        sentiment_list = []
        print("loading submission")
        submissions_df = pd.read_pickle(FILTERED_SUBMISSIONS_FILENAME)
        author_list.append(submissions_df["author"])
        timestamp_list.append(submissions_df["timestamp"])
        print("loading comments")
        for month in tqdm(MONTHS):
            comments_df = pd.read_pickle("{}language-features-{}-{}.p".format(FEATURE_BASE_DIR, YEAR, month))
            author_list.append(comments_df["author"])
            timestamp_list.append(comments_df["timestamp"])
            sentiment_list.append(comments_df["sentiment"])
        print("creating series")
        author_s = pd.concat(author_list)
        timestamp_s = pd.concat(timestamp_list)
        sentiment_s = pd.concat(sentiment_list)
        print("loading graph")
        g = ig.Graph.Read_Pickle(FILTERED_GRAPH_FILENAME)
        print("adding attributes")
        g.vs["author"] = author_s.reindex(g.vs["name"]).values
        g.vs["timestamp"] = timestamp_s.reindex(g.vs["name"]).values
        g.vs["sentiment"] = sentiment_s.reindex(g.vs["name"]).values
        for month in MONTHS:
            print("month {}: processing comments".format(month))
            comments_df = pd.read_pickle("{}language-features-{}-{}.p".format(FEATURE_BASE_DIR, YEAR, month))
            structural_features_raw = []
            for row in tqdm(comments_df.itertuples(), total=len(comments_df)):
                cmt_id = row[0]
                predecessors = g.neighborhood(cmt_id, mode="in", order=len(g.es))
                successors = g.neighborhood(cmt_id, mode="out", order=len(g.es))[1:]
                num_predecessors = len(predecessors)
                num_successors = len(successors)
                this_ts = g.vs[predecessors.pop(0)]["timestamp"]
                predecessors_vs = ig.VertexSeq(g, predecessors)
                successors_vs = ig.VertexSeq(g, successors)
                predecessors_ts = predecessors_vs["timestamp"]
                if None in predecessors_ts:
                    seconds_to_predecessor = None
                    mean_predecessors_frequency = None
                    num_predecessors_unique_authors = None
                    mean_predecessors_sentiment = None
                else:
                    predecessors_at = predecessors_vs["author"]
                    predecessors_se = np.array(predecessors_vs["sentiment"])
                    seconds_to_predecessor = this_ts - max(predecessors_ts)
                    if num_predecessors > 1:
                        mean_predecessors_frequency = np.mean(np.abs(np.diff(predecessors_ts)))
                    else:
                        mean_predecessors_frequency = seconds_to_predecessor
                    mean_predecessors_sentiment = np.mean(predecessors_se[~np.isnan(predecessors_se)])
                    num_predecessors_unique_authors = len(np.unique(predecessors_at)) 
                if num_successors > 0:
                    successors_ts = successors_vs["timestamp"]
                    seconds_to_successor = min(successors_ts) - this_ts
                    if num_successors > 1:    
                        mean_successors_frequency = np.mean(np.diff(successors_ts))
                    else:
                        mean_successors_frequency = seconds_to_successor
                    successors_at = successors_vs["author"]
                    num_successors_unique_authors = len(np.unique(successors_at))
                    successors_se = np.array(successors_vs["sentiment"])
                    mean_successors_sentiment = np.mean(successors_se[~np.isnan(successors_se)])
                else:
                    seconds_to_successor = None
                    num_successors_unique_authors = None
                    mean_successors_sentiment = None 
                    mean_successors_frequency = None
                structural_features_raw.append({
                                                    "num_predecessors": num_predecessors,
                                                    "seconds_to_predecessor": seconds_to_predecessor,
                                                    "mean_predecessors_frequency": mean_predecessors_frequency,
                                                    "num_predecessors_unique_authors": num_predecessors_unique_authors,
                                                    "mean_predecessors_sentiment": mean_predecessors_sentiment,
                                                    "num_successors": num_successors,
                                                    "seconds_to_successor": seconds_to_successor,
                                                    "mean_successor_frequency": mean_successors_frequency,
                                                    "num_successors_unique_authors": num_successors_unique_authors,
                                                    "mean_successors_sentiment": mean_successors_sentiment
                                                })
            structural_features_df = pd.DataFrame(structural_features_raw, index=comments_df.index)
            result_df = pd.concat([comments_df, structural_features_df], axis=1)
            print("month {}: saving features".format(month))
            result_df.to_pickle("{}structural-features-{}-{}.p".format(FEATURE_BASE_DIR, YEAR, month))
        print("done")

    if COMPUTE_TOP_WORDS:
        print("loading data")
        combined_raw = []
        for month in tqdm(MONTHS, desc="loading features"):
            features_df = pd.read_pickle("{}all-features-{}-{}.p".format(FEATURE_BASE_DIR, YEAR, month))
            features_df["month"] = month
            combined_raw.append(features_df)
        
        print("combining data")
        combined_df = pd.concat(combined_raw)

        print("removing deleted comments", end="")
        num_before = len(combined_df)
        combined_df = combined_df[~combined_df["text"].isin(["[deleted]", "[removed]"])]
        num_deleted = num_before - len(combined_df)
        print(" (removed {} comments)".format(num_deleted))

        print("removing comments with language discrepancies", end="")
        num_before = len(combined_df)
        combined_df = combined_df[combined_df["subreddit_language"] == combined_df["detected_language"]]
        num_deleted = num_before - len(combined_df)
        print(" (removed {} comments)".format(num_deleted))

        print("computing content features")
        extracted_top_words = {}
        for language in tqdm(LANGUAGES):
            l_df = combined_df[(combined_df["subreddit_language"] == language)].copy()
            c_df = l_df[l_df["controversial"] == True]
            nc_df = l_df[l_df["controversial"] == False]
            if language == "en":
                tp = r"[A-Za-zÀ-ȕ']{3,}"
            else:
                tp = r"[A-Za-zÀ-ȕ]{3,}"
            vectorizer = CountVectorizer(stop_words=get_stop_words(language), token_pattern=tp)
            tf = vectorizer.fit_transform([" ".join(c_df["text"].tolist()), " ".join(nc_df["text"].tolist())])
            tf_df = pd.DataFrame(tf.toarray(), index=["c_count", "nc_count"], columns=vectorizer.get_feature_names()).T
            top_c_w = tf_df["c_count"].sort_values(ascending=False).head(500).index.tolist()
            top_nc_w = tf_df["nc_count"].sort_values(ascending=False).head(500).index.tolist()
            union_w = set(top_c_w + top_nc_w)
            print(language, len(union_w))
            s_result_df = tf_df[tf_df.index.isin(union_w)].copy()
            s_result_df["controversial"] = np.where(s_result_df.index.isin(top_c_w), True, False)
            s_result_df["non_controversial"] = np.where(s_result_df.index.isin(top_nc_w), True, False)
            for w, row in s_result_df.iterrows():
                not_w_count_c = s_result_df["c_count"].sum() - row["c_count"]
                not_w_count_nc = s_result_df["nc_count"].sum() - row["nc_count"]
                con_table = [[row["c_count"], row["nc_count"]], [not_w_count_c, not_w_count_nc]]
                chi2, p, dof, ex = chi2_contingency(con_table, correction=True)
                s_result_df.at[w, "chi2"] = chi2
                s_result_df.at[w, "p"] = p
            top_c_words = {}
            top_nc_words = {}
            num_top_words = 25
            for idx, row in s_result_df.sort_values("chi2", ascending=False).iterrows():
                c_rel_count = row["c_count"] / s_result_df["c_count"].sum()
                nc_rel_count = row["nc_count"] / s_result_df["nc_count"].sum()
                if c_rel_count > nc_rel_count and len(top_c_words) < num_top_words:
                    top_c_words[idx] = int(row["chi2"])
                if nc_rel_count > c_rel_count and len(top_nc_words) < num_top_words:
                    top_nc_words[idx] = int(row["chi2"])
                if len(top_c_words) == num_top_words and len(top_nc_words) == num_top_words:
                    break
            extracted_top_words[language] = {"controversial": top_c_words, "non_controversial": top_nc_words} 

        print("saving top words")
        with open("results/top_words.p", "wb") as handle:
            pickle.dump(extracted_top_words, handle)
        print("done")

    if COMPUTE_CONTENT_FEATURES:
        print("loading top words")
        with open("results/top_words.p", "rb") as handle:
            top_words = pickle.load(handle)

        def process_month(month):
            print("month {}: starting".format(month))
            comments_df = pd.read_pickle("{}structural-features-{}-{}.p".format(FEATURE_BASE_DIR, YEAR, month))
            text_index = list(comments_df.columns).index("text") + 1
            language_index = list(comments_df.columns).index("subreddit_language") + 1
            new_features_raw = []
            for row in comments_df.itertuples():
                c_top_words = top_words[row[language_index]]["controversial"].keys()
                nc_top_words = top_words[row[language_index]]["non_controversial"].keys()
                num_controversial_words = 0
                num_non_controversial_words = 0
                for w in c_top_words:
                    num_controversial_words += row[text_index].count(w)
                for w in nc_top_words:
                    num_non_controversial_words += row[text_index].count(w)
                new_features_raw.append({"num_controversial_words": num_controversial_words, "num_non_controversial_words": num_non_controversial_words})
            print("month {}: creating data frame".format(month))
            new_features_df = pd.DataFrame(new_features_raw, index=comments_df.index)
            print("month {}: appending dataframe".format(month))
            comments_df = pd.concat([comments_df, new_features_df], axis=1)
            print("saving new dataframe")
            comments_df.to_pickle("{}all-features-{}-{}.p".format(FEATURE_BASE_DIR, YEAR, month))
            print("month {}: done".format(month))

        print("computing content features")
        Parallel(n_jobs=MAX_PROCESSES)(delayed(process_month)(month) for month in MONTHS)
        print("done")

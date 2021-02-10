import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.patches import Patch

plt.style.use("seaborn-whitegrid")
plt.rc('ps',fonttype = 42)
plt.rc('pdf',fonttype = 42)
plt.rcParams.update({'font.size': 20})
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['axes.unicode_minus'] = False

YEAR = 2019
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
LANGUAGES = ["en", "fr", "de", "it", "pt", "es"]
CMT_DF_BASE_DIR = "data/processed/comments/"
FEATURE_BASE_DIR = "data/processed/features/"
PLOT_RESULTS_DIR = "results/plots/"

LANGUAGE_MAPPING = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish"
}

FEATURE_NAME_MAPPING = {
    "num_controversial_words": "Number of\nControversial Words",
    "num_non_controversial_words": "Number of Non-\nControversial Words",
    "num_chars": "Number of\nCharacters",
    "num_syllables": "Number of\nSyllables",
    "num_words": "Number of\nWords", 
    "num_sentences": "Number of\nSentences", 
    "characters_to_sentences_ratio": "Characters to\nSentences Ratio",
    "words_to_sentences_ratio": "Words to\nSentences Ratio", 
    "flesch_reading_ease": "Flesch\nReading Ease", 
    "pos_ratio_nouns": "Ratio of Nouns",
    "pos_ratio_verbs": "Ratio of Verbs",
    "pos_ratio_adjectives": "Ratio of Adjectives", 
    "pos_ratio_adverbs": "Ratio of Adverbs",
    "pos_ratio_pronouns": "Ratio of Pronouns",
    "sentiment": "Sentiment",
    "mean_predecessors_sentiment": "Preceding\nMean Sentiment",
    "mean_successors_sentiment": "Succeeding\nMean Sentiment",
    "num_predecessors": "Number of\nPredecessors", 
    "num_successors": "Number of\nSuccessors",
    "num_predecessors_unique_authors": "Unique\nPreceding Users",
    "num_successors_unique_authors": "Unique\nSucceeding Users",
    "seconds_to_predecessor": "Time From\nPredecessor",
    "seconds_to_successor": "Time To\nFirst Successor", 
    "mean_predecessors_frequency": "Mean Time Between\nPredecessors",
    "mean_successor_frequency": "Mean Time Between\nSuccessors",
}

FEATURE_COLOR_MAPPING = {
    "num_controversial_words": sns.color_palette("Set2")[0],
    "num_non_controversial_words": sns.color_palette("Set2")[0],
    "num_chars": sns.color_palette("Set2")[1],
    "num_syllables": sns.color_palette("Set2")[1],
    "num_words": sns.color_palette("Set2")[1], 
    "num_sentences": sns.color_palette("Set2")[1], 
    "characters_to_sentences_ratio": sns.color_palette("Set2")[1],
    "words_to_sentences_ratio": sns.color_palette("Set2")[1], 
    "flesch_reading_ease": sns.color_palette("Set2")[1], 
    "pos_ratio_nouns": sns.color_palette("Set2")[1],
    "pos_ratio_verbs": sns.color_palette("Set2")[1],
    "pos_ratio_adjectives": sns.color_palette("Set2")[1], 
    "pos_ratio_adverbs": sns.color_palette("Set2")[1],
    "pos_ratio_pronouns": sns.color_palette("Set2")[1],
    "sentiment": sns.color_palette("Set2")[2],
    "mean_predecessors_sentiment": sns.color_palette("Set2")[2],
    "mean_successors_sentiment": sns.color_palette("Set2")[2],
    "num_predecessors": sns.color_palette("Set2")[3], 
    "num_successors": sns.color_palette("Set2")[3],
    "num_predecessors_unique_authors": sns.color_palette("Set2")[3],
    "num_successors_unique_authors": sns.color_palette("Set2")[3],
    "seconds_to_predecessor": sns.color_palette("Set2")[3],
    "seconds_to_successor": sns.color_palette("Set2")[3], 
    "mean_predecessors_frequency": sns.color_palette("Set2")[3],
    "mean_successor_frequency": sns.color_palette("Set2")[3],
}

if __name__ == "__main__":
    print("training model")
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

    print("removing comments with less than 100 words", end="")
    num_before = len(combined_df)
    combined_df = combined_df[combined_df["num_words"] >= 100]
    num_deleted = num_before - len(combined_df)
    print(" (removed {} comments)".format(num_deleted))
  
    print("final number of comments:", len(combined_df))
    print("final number of controversial comments:", len(combined_df[combined_df["controversial"] == True]))
    print("final number of non-controversial comments:", len(combined_df[combined_df["controversial"] == False]))
    for language in LANGUAGES:
        print("final number of {} comments:".format(language), combined_df[combined_df["subreddit_language"] == language]["controversial"].value_counts())
    
    print("preparing data")
    X = combined_df.drop(["text", "type", "timestamp", "author", "submission_id", "parent_id", "score", "month"], axis=1)
    X.dropna(inplace=True)
    y = X.pop("controversial")

    enc = OneHotEncoder(handle_unknown='ignore')

    cat_X = X.pop("subreddit")
    lang_X = X.pop("subreddit_language")
    cat_X = pd.DataFrame(enc.fit_transform(cat_X.values.reshape(-1, 1)).todense(), index=X.index)
    subreddit_columns = ["Posted in\nr/" + sr for sr in enc.categories_[0].tolist()]
    lang_X = pd.DataFrame(enc.fit_transform(lang_X.values.reshape(-1, 1)).todense(), index=X.index)
    language_columns = ["Posted in " + LANGUAGE_MAPPING[l] for l in enc.categories_[0].tolist()]

    X.drop(["detected_language"], inplace=True, axis=1)
    X_ui = X[["num_predecessors", "num_successors", "num_predecessors_unique_authors", "num_successors_unique_authors", "seconds_to_predecessor", "seconds_to_successor", "mean_predecessors_frequency", "mean_successor_frequency"]].copy()
   
    X = X[FEATURE_NAME_MAPPING.keys()]

    feature_columns = [FEATURE_NAME_MAPPING[cn] for cn in X.columns]

    col_pal = {FEATURE_NAME_MAPPING[k]:v for k, v in FEATURE_COLOR_MAPPING.items()}

    # for c in language_columns:
    #     col_pal[c] = sns.color_palette("Set2")[4]

    for c in subreddit_columns:
        col_pal[c] = sns.color_palette("Set2")[4]

    X = pd.DataFrame(RobustScaler().fit_transform(X), index=X.index)
    X_ui = pd.DataFrame(RobustScaler().fit_transform(X_ui), index=X_ui.index)

    X.columns = range(0, len(X.columns))
    X_ui.columns = range(0, len(X_ui.columns))
    lang_X.columns = range(50, len(lang_X.columns) + 50)
    cat_X.columns = range(100, len(cat_X.columns) + 100)

    print(X.shape)
    print(cat_X.shape)
    print(lang_X.shape)

    X_resampled, y_resampled = RandomUnderSampler().fit_resample(pd.concat([X, lang_X, cat_X], axis=1), y)
    X_ui_resampled, y_ui_resampled = RandomUnderSampler().fit_resample(pd.concat([X_ui, lang_X, cat_X], axis=1), y)
    #X_wuwise_resampled, y_wuwise_resampled = RandomUnderSampler().fit_resample(pd.concat([X_wuwise, lang_X, cat_X], axis=1), y)

    print(X_resampled.shape)

    # grid search (this was only used to find optimal parameters)
    # print("conducting grid search")
    # parameters = {
    #     "max_depth": [5, 6, 7, 8, 9],
    #     "min_samples_split": [2, 3, 4, 5],
    #     "min_samples_leaf": [1, 2, 3, 4, 5],
    #     "subsample": [0.95, 0.9, 0.85]
    # }
    # gbc = GradientBoostingClassifier()
    # clf = GridSearchCV(gbc, parameters, n_jobs=22, scoring="roc_auc")
    # clf.fit(X_resampled, y_resampled)
    # print(clf.best_score_)
    # print(clf.best_params_)
    # exit()

    print("cross validating real model")
    clf = GradientBoostingClassifier(max_depth=7, min_samples_leaf=5, min_samples_split=2, subsample=0.95)

    cv_ui_results = cross_validate(clf, X_ui_resampled, y_ui_resampled, cv=10, return_estimator=True, n_jobs=10, scoring="roc_auc")
    print("mean test_score:", np.mean(cv_ui_results["test_score"]))
    cv_wuwise_results = cross_validate(clf, X_wuwise_resampled, y_wuwise_resampled, cv=10, return_estimator=True, n_jobs=10, scoring="roc_auc")
    print("mean test_score:", np.mean(cv_wuwise_results["test_score"]))

    cv_results = cross_validate(clf, X_resampled, y_resampled, cv=10, return_estimator=True, n_jobs=10, scoring="roc_auc")
    print("mean test_score:", np.mean(cv_results["test_score"]))
    print("saving models")
    with open("results/prediction_models.p", "wb") as handle:
        pickle.dump(cv_results["estimator"], handle)

    print("plottig feature importances")
    f_i_raw = []
    for num, model in enumerate(cv_results["estimator"]):
        f_i_s = pd.Series()
        f_i = model.feature_importances_
        for val, feat in zip(f_i, feature_columns + language_columns + subreddit_columns):
            if feat not in f_i_s:
                f_i_s[feat] = val
            else:
                f_i_s[feat] += val
        temp_df = f_i_s.to_frame().reset_index()
        temp_df.columns = ["feature", "value"]
        temp_df["num"] = num
        f_i_raw.append(temp_df)
    f_i_df = pd.concat(f_i_raw)
    
    sort_df = f_i_df.groupby("feature")["value"].mean()
    idx_to_keep = sort_df[sort_df.round(2) > 0].index
    f_i_df = f_i_df [f_i_df["feature"].isin(idx_to_keep)]

    def show_values_on_bars(axs):
        def _show_on_single_plot(ax):        
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + 0.02
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center", fontsize=10) 

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)

    fig, ax = plt.subplots(figsize=(12,2.5))
    sns.barplot(data=f_i_df, x="feature", y="value", ax=ax, palette=col_pal)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=90, ma="right")
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels([0, 0.1, 0.2, 0.3], fontsize=12)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Feature\nImportance", fontsize=12)
    show_values_on_bars(ax)
    sns.despine(fig)
    plt.subplots_adjust(left=0.06, bottom=0.70, right=0.99, top=0.95)
    plt.savefig("{}feature_importances.pdf".format(PLOT_RESULTS_DIR))
    plt.close()

    exit()

    print("plotting legend")
    fig, ax = plt.subplots(figsize=(10,0.25))
    handles = [
        Patch(facecolor=sns.color_palette("Set2")[0], label="Word Usage"),
        Patch(facecolor=sns.color_palette("Set2")[1], label="Writing Style"),
        Patch(facecolor=sns.color_palette("Set2")[2], label="Sentiment"),
        Patch(facecolor=sns.color_palette("Set2")[3], label="User Involvement"),
        #Patch(facecolor=sns.color_palette("Set2")[4], label="Language"),
        Patch(facecolor=sns.color_palette("Set2")[4], label="Subreddit"),
    ]
    ax.legend(handles=handles, loc='center', ncol=6, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("results/plots/prediction_legend.pdf")
    plt.close()

    # Stratified Random Baseline
    # print("cross validating stratified random baseline")
    # dummy_clf = DummyClassifier(strategy="stratified")
    # dummy_cv_results = cross_validate(dummy_clf, X_resampled, y_resampled, cv=10, n_jobs=10, scoring="roc_auc")
    # print("baseline score:", np.mean(dummy_cv_results["test_score"]))

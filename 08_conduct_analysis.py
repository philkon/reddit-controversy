import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from wordcloud import WordCloud
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from scipy.stats import mannwhitneyu, levene, median_test, pearsonr
from matplotlib.patches import Patch
from itertools import combinations

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
    #"num_controversial_words": "Number of Controversial Words",
    #"num_non_controversial_words": "Number of Non-Controversial Words",
    "num_chars": "Number of Characters",
    "num_syllables": "Number of Syllables",
    "num_words": "Number of Words", 
    "num_sentences": "Number of Sentences", 
    "characters_to_sentences_ratio": "Characters to Sentences Ratio",
    "words_to_sentences_ratio": "Words to Sentences Ratio", 
    "flesch_reading_ease": "Flesch Reading Ease", 
    "pos_ratio_nouns": "Ratio of Nouns",
    "pos_ratio_verbs": "Ratio of Verbs",
    "pos_ratio_adjectives": "Ratio of Adjectives", 
    "pos_ratio_adverbs": "Ratio of Adverbs",
    "pos_ratio_pronouns": "Ratio of Pronouns",
    "sentiment": "Sentiment",
    "mean_predecessors_sentiment": "Preceding Mean Sentiment",
    "mean_successors_sentiment": "Succeeding Mean Sentiment",
    "num_predecessors": "Number of Predecessors", 
    "num_successors": "Number of Successors",
    "num_predecessors_unique_authors": "Unique Preceding Users",
    "num_successors_unique_authors": "Unique Succeeding Users",
    "seconds_to_predecessor": "Seconds From Predecessor",
    "seconds_to_successor": "Seconds To First Successor", 
    "mean_predecessors_frequency": "Mean Predecessors Frequency",
    "mean_successor_frequency": "Mean Successors Frequency",
}

LINE_TYPES = {
    "en": "solid",
    "fr": "dotted",
    "de": "dashed",
    "it": "dashdot",
    "pt": "solid",
    "es": "dotted"
}

MIN_WORDS_FEATURES = [
    "flesch_reading_ease",
    "sentiment"
]

def preliminary_plot_submission_length():
    print("plotting submission length...", end="")
    fig, axs = plt.subplots(1, 6, sharex=True, sharey=True, figsize=(10, 2))
    lower, upper = np.percentile(combined_df.groupby("submission_id").size(), [5, 95])
    print(lower, upper, "...")
    for ax_idx, language in enumerate(LANGUAGES):
        submission_length = combined_df[combined_df["subreddit_language"] == language].groupby("submission_id").size()
        lower, upper = np.percentile(submission_length, [5, 95])
        submission_length = submission_length[submission_length < upper]
        sns.kdeplot(submission_length, ax=axs[ax_idx])
        axs[ax_idx].set_title(LANGUAGE_MAPPING[language], fontsize=20)
        if ax_idx == 0:
            axs[ax_idx].set_ylabel("Density")
    sns.despine(fig)
    fig.text(0.5, 0.04, "Number of Comments", ha="center")
    plt.subplots_adjust(left=0.1, bottom=0.30, right=0.99, top=0.8)
    plt.savefig("results/plots/preliminary_submission_length.pdf")
    plt.close()
    print("done")
    
def preliminary_plot_score():
    print("plotting score...", end="")
    fig, axs = plt.subplots(1, 6, sharex=True, sharey=True, figsize=(10, 2))
    lower, upper = np.percentile(combined_df["score"], [5, 95])
    print(lower, upper, "...")
    for controversial in [True, False]:
        if controversial:
            color = "red"
        else:
            color = "green"
        for ax_idx, language in enumerate(LANGUAGES):
            score = combined_df[(combined_df["subreddit_language"] == language) & (combined_df["controversial"] == controversial)]["score"]
            lower, upper = np.percentile(score, [5, 95])
            score = score[score < upper]
            score = score[score > lower]
            sns.kdeplot(score, ax=axs[ax_idx], color=color, bw_adjust=2.5)
            axs[ax_idx].set_xlabel(None)
            if controversial:
                axs[ax_idx].set_title(LANGUAGE_MAPPING[language], fontsize=20)
            if ax_idx == 0 and controversial:
                axs[ax_idx].set_ylabel("Density")
    sns.despine(fig)
    fig.text(0.5, 0.04, "Comment Score", ha="center")
    plt.subplots_adjust(left=0.1, bottom=0.30, right=0.99, top=0.8)
    plt.savefig("results/plots/preliminary_score.pdf")
    plt.close()
    print("done")

def preliminary_print_submission_length():
    print("mean submission length:", combined_df.groupby("submission_id").size().mean().round(2))
    print("median submission length:", combined_df.groupby("submission_id").size().median().round(2))

def preliminary_print_score():
    print("mean score (non-controversial):", combined_df[combined_df["controversial"] == False]["score"].mean().round(2))
    print("median score (non-controversial):", combined_df[combined_df["controversial"] == False]["score"].median().round(2))
    print("mean score (controversial):", combined_df[combined_df["controversial"] == True]["score"].mean().round(2))
    print("median score (controversial):", combined_df[combined_df["controversial"] == True]["score"].median().round(2))

def preliminary_controversial_ranking():
    print("creating controversial ranking table...", end="")
    ratios_df = combined_df.groupby(["subreddit_language", "subreddit"])["controversial"].sum() / combined_df.groupby(["subreddit_language", "subreddit"]).size()
    ratios_df = ratios_df.to_frame().reset_index()
    ratios_df.columns = ["language", "subreddit", "ratio"]
    result_raw = []
    for language in LANGUAGES:
        language_df = ratios_df[ratios_df["language"] == language].sort_values("ratio", ascending=False)
        language_df = language_df.reset_index()
        language_df["subreddit"] = language_df["subreddit"].apply(lambda x: "r/" + x.replace("_", "\_"))
        language_df["ratio"] = language_df["ratio"].apply(lambda x: r"${:1.2f}\%$".format(round(x * 100, 2)))
        result_raw.append(language_df[["subreddit", "ratio"]])
        #break
    result_df = pd.concat(result_raw, axis=1)
    result_df.to_latex(buf="results/subreddit_controversial_statistics.txt", header=False, index=False, escape=False, na_rep="")
    print("done")


def create_box_plot(df, feature_name):
    print("creating box plot for {}...".format(feature_name), end="")
    if feature_name in MIN_WORDS_FEATURES:
        plot_df = df[df["num_words"] >= 100]
    else:
        plot_df = df
    if feature_name == "automated_readability_index":
        #plot_df["automated_readability_index"] = plot_df["automated_readability_index"].apply(lambda x: 20 if x > 20 else x)
        plot_df["automated_readability_index"] = plot_df["automated_readability_index"].apply(lambda x: 0 if x < 0 else x)
    plot_df = plot_df.dropna(subset=[feature_name])
    my_pal = {True: "r", False: "g"}
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="subreddit_language", y=feature_name, hue="controversial", data=plot_df, showfliers=False, palette=my_pal, boxprops=dict(alpha=.8), order=LANGUAGE_MAPPING.keys(), ax=ax, hue_order=[True, False])
    ax.set(xlabel="Language", ylabel=FEATURE_NAME_MAPPING[feature_name])
    ax.legend_.remove()
    ax.set_xticklabels(LANGUAGE_MAPPING.values(), rotation=15, fontsize=25)
    sns.despine(fig)
    plt.tight_layout()
    plt.savefig("{}boxplot_{}.pdf".format(PLOT_RESULTS_DIR, feature_name))
    plt.close()
    print("done")

def plot_box_plot_legend():
    print("plotting box plot legend...", end="")
    handles = [
        Patch(facecolor="r", label="Controversial Comments", alpha=0.8),
        Patch(facecolor="g", label="Non-Controversial Comments", alpha=0.8),
    ]

    fig, ax = plt.subplots(figsize=(10,0.25))
    ax.legend(handles=handles, loc='center', ncol=2, fontsize=10)
    ax.axis("off")
    plt.savefig("{}boxplot_legend.pdf".format(PLOT_RESULTS_DIR))
    plt.close()
    print("done")

def create_temporal_plot(df, feature_name):
    print("creating temporal plot for {}...".format(feature_name), end="")
    if feature_name in MIN_WORDS_FEATURES:
        plot_df = df[df["num_words"] >= 50]
    else:
        plot_df = df
    plot_df = plot_df.dropna(subset=[feature_name])
    plt.figure(figsize=(10,5))
    ax = sns.lineplot(x="month", y=feature_name, hue="detected_language", data=plot_df)
    #ax.set(xlabel="Language", ylabel=FEATURE_MAPPING[feature_name])
    #ax.legend_.remove()
    plt.savefig("{}temporal_{}.pdf".format(PLOT_RESULTS_DIR, feature_name))
    plt.close()
    print("done")

def print_top_words():
    print("printing top words")
    with open("results/top_words.p", "rb") as handle:
        top_words = pickle.load(handle)

    for language in LANGUAGES:
        print(language)
        print(", ".join(top_words[language]["controversial"].keys()))
        print(", ".join(top_words[language]["non_controversial"].keys()))
    print("done")

def plot_top_words():
    print("plotting top words")
    with open("results/top_words.p", "rb") as handle:
        top_words = pickle.load(handle)

    for language in tqdm(LANGUAGES):
        c_words = top_words[language]["controversial"]
        nc_words = top_words[language]["non_controversial"]

        def get_wordcloud_color(word, font_size, position, orientation, random_state, font_path):
            if word in c_words:
                return "red"
            if word in nc_words:
                return "green"

        combined_words = {**c_words, **nc_words}
 
        wordcloud = WordCloud(font_path="helvetica.ttf", width=1000,height=500, background_color="white", color_func=get_wordcloud_color, relative_scaling=0, min_font_size=20, max_font_size=100).generate_from_frequencies(combined_words)
        with open("results/plots/top_words_{}.svg".format(language), "w") as svg_file:
            svg_file.write(wordcloud.to_svg())

        drawing = svg2rlg("results/plots/top_words_{}.svg".format(language))
        renderPDF.drawToFile(drawing, "results/plots/top_words_{}.pdf".format(language))
    print("done")

def print_num_over_100():
    for language in LANGUAGES:
        language_df = combined_df[combined_df["subreddit_language"] == language]
        over_df = language_df[language_df["num_words"] >= 100]
        print(language, len(over_df))


def compute_p_values():
    df_index = pd.MultiIndex.from_product([LANGUAGES, ["variance", "mediantest", "utest", "median_dif", "mean_dif"]], names=["language", "type"])
    pvalues_df = pd.DataFrame(index=df_index)
    for col in tqdm(FEATURE_NAME_MAPPING.keys()):
        if col in ["num_controversial_words", "num_non_controversial_words"]:
            continue
        for language in LANGUAGES:
            language_df = combined_df[combined_df["subreddit_language"] == language]
            if col in MIN_WORDS_FEATURES:
                language_df = language_df[language_df["num_words"] >= 100]
            controversial_col = language_df[language_df["controversial"] == True][col].dropna().values
            non_controversial_col = language_df[language_df["controversial"] == False][col].dropna().values
            controversial_median = np.median(controversial_col)
            non_controversial_median = np.median(non_controversial_col)
            controversial_mean = np.mean(controversial_col)
            non_controversial_mean = np.mean(non_controversial_col)
            median_dif = controversial_median - non_controversial_median
            mean_dif = controversial_mean - non_controversial_mean
            variance = levene(controversial_col, non_controversial_col, center="median").pvalue
            mediantest = median_test(controversial_col, non_controversial_col)[1]
            utest = mannwhitneyu(controversial_col, non_controversial_col).pvalue
            pvalues_df.at[(language, "median_dif"), col] = median_dif
            pvalues_df.at[(language, "mean_dif"), col] = mean_dif
            pvalues_df.at[(language, "variance"), col] = variance
            pvalues_df.at[(language, "mediantest"), col] = mediantest
            pvalues_df.at[(language, "utest"), col] = utest
    significant = 0
    total = 0
    for col in pvalues_df.columns.tolist():
        for language in LANGUAGES:
            language_variance_raw = pvalues_df.loc[(language, "variance"), col]
            language_mediantest_raw = pvalues_df.loc[(language, "mediantest"), col]
            language_utest_raw = pvalues_df.loc[(language, "utest"), col]
            if language_variance_raw < 0.05:
                language_pvalue = language_mediantest_raw
            else:
                language_pvalue = language_utest_raw
            if language_pvalue < (0.05/23):
                significant += 1
            total += 1
            print(col, language, language_pvalue)
    print(significant, total)

    print()

    for col in pvalues_df.columns.tolist():
        print(FEATURE_NAME_MAPPING[col], end=" & ")
        for sr in LANGUAGES:
            sr_median_raw = pvalues_df.loc[(sr, "median_dif"), col]
            sr_mean_raw = pvalues_df.loc[(sr, "mean_dif"), col]
            sr_variance_raw = pvalues_df.loc[(sr, "variance"), col]
            sr_mediantest_raw = pvalues_df.loc[(sr, "mediantest"), col]
            sr_utest_raw = pvalues_df.loc[(sr, "utest"), col]
            if sr_variance_raw < 0.05:
                sr_pvalue = sr_mediantest_raw
                median_mark = ""
            else:
                sr_pvalue = sr_utest_raw
                median_mark = "\cellcolor{blue!25}"
            if sr_pvalue < (0.05/23):
                #sr_pvalue = "<\\alpha"
                mark1 = "\\underline{"
                mark2 = "}"
            else:
                sr_pvalue = round(sr_pvalue, 4)
                mark1 = ""
                mark2 = ""
            sr_value = "${}{}{}\:({}){}$".format(median_mark, mark1, round(sr_median_raw, 3), round(sr_mean_raw, 3), mark2)
            #if "e" in str(sr_pvalue):
            #    sr_split = str(sr_pvalue).split("e")
            #    sr_value = "${}{}{} \\times 10^{{{}}}{}\:({:.2f})$".format(median_mark, mark1, round(float(sr_split[0]), 2), sr_split[1], mark2, round(sr_median_raw, 2))
            #else:
            #    sr_value = "${}{}{}{}\:({:.2f})$".format(median_mark, mark1, round(sr_pvalue, 2), mark2, round(sr_median_raw, 2))
            if sr == "es":
                print(sr_value, end=" \\\\\n")
            else:
                print(sr_value, end=" & ")

def compute_corr_comment_length():
    print("computing correlation coefficients")
    rhos = []
    for col1, col2 in combinations(["num_chars", "num_syllables", "num_words", "num_sentences"], 2):
        rho, p = pearsonr(combined_df[col1], combined_df[col2])
        print(col1, col2, rho, p)
        rhos.append(rho)
    print("mean:", np.mean(rhos))

def compute_corr_temporal_structural():
    print("computing correlation coefficients")
    print(combined_df[["num_successors", "seconds_to_successor"]].corr(method="spearman"))
    #for col1, col2 in combinations(["num_successors", "seconds_to_successor"], 2):
    #    rho, p = pearsonr(combined_df[col1].dropna(), combined_df[col2].dropna())
    #    print(col1, col2, rho, p)

def compute_corr_num_cmts_num_cont():
    #for language in LANGUAGES:
    #    language_df = combined_df[combined_df["subreddit_language"] == language]
    #    num_comments = language_df.groupby(["subreddit", "submission_id"]).size()
        #print(num_comments)
    #    num_controversial_comments = language_df[language_df["controversial"] == True].groupby(["subreddit", "submission_id"]).size().reindex_like(num_comments).fillna(0)
        #print(num_controversial_comments)
    #    print(language, pearsonr(num_comments, num_controversial_comments))
    for sr in combined_df["subreddit"].unique():
    #    print(sr)
        sr_df = combined_df[combined_df["subreddit"] == sr]
        num_comments = sr_df.groupby("submission_id").size()
        num_controversial_comments = sr_df[sr_df["controversial"] == True].groupby("submission_id").size().reindex_like(num_comments).fillna(0)
        print(sr, pearsonr(num_comments, num_controversial_comments))
    #    print(sr, pearsonr(len(sr_df), len(sr_df[sr_df["controversial"] == True])))

def print_subreddit_sizes():
    print(combined_df[combined_df["controversial"] == True].groupby("subreddit").size().sort_values())#.to_csv("results/subreddit_sizes.csv")
    print(combined_df[combined_df["controversial"] == False].groupby("subreddit").size().sort_values())

if __name__ == "__main__":
    print("loading features")
    combined_raw = []
    for month in tqdm(MONTHS):
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

    # preliminary_print_submission_length()
    # preliminary_print_score()
    # preliminary_plot_submission_length()
    # preliminary_plot_score()
    # preliminary_controversial_ranking()
    # print_num_over_100()
    # create_box_plot(combined_df, "num_chars")
    # create_box_plot(combined_df, "num_words")
    # create_box_plot(combined_df, "num_sentences")
    # create_box_plot(combined_df, "characters_to_sentences_ratio")
    # create_box_plot(combined_df, "words_to_sentences_ratio")
    # create_box_plot(combined_df, "automated_readability_index")
    # create_box_plot(combined_df, "coleman_liau_index")
    # create_box_plot(combined_df, "dale_chall")
    # create_box_plot(combined_df, "flesch_reading_ease")
    # create_box_plot(combined_df, "flesch_kincaid_grade_level")
    # create_box_plot(combined_df, "forcast")
    # create_box_plot(combined_df, "smog")
    # create_box_plot(combined_df, "pos_ratio_nouns")
    # create_box_plot(combined_df, "pos_ratio_verbs")
    # create_box_plot(combined_df, "pos_ratio_adjectives")
    # create_box_plot(combined_df, "pos_ratio_adverbs")
    # create_box_plot(combined_df, "pos_ratio_pronouns")
    # plot_box_plot_legend()
    # create_box_plot(combined_df, "sentiment")
    # create_box_plot(combined_df, "mean_predecessors_sentiment")
    # create_box_plot(combined_df, "mean_successors_sentiment")
    # create_box_plot(combined_df, "num_predecessors")
    # create_box_plot(combined_df, "num_successors")
    # create_box_plot(combined_df, "num_predecessors_unique_authors")
    # create_box_plot(combined_df, "num_successors_unique_authors")
    # create_box_plot(combined_df, "seconds_to_predecessor")
    # create_box_plot(combined_df, "seconds_to_successor")
    # create_box_plot(combined_df, "mean_predecessors_frequency")
    # create_box_plot(combined_df, "mean_successor_frequency")
    # compute_p_values()
    # compute_corr_comment_length()
    # compute_corr_temporal_structural()
    # compute_corr_num_cmts_num_cont()
    # print_subreddit_sizes()

    print("done")

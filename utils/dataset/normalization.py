from nltk.tokenize import TweetTokenizer
from emoji import demojize
import pandas as pd
import re
import unicodedata

tokenizer = TweetTokenizer()


def normalize_token(token) -> str:
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalize_text(norm_tweet, to_ascii=True, to_lower=False) -> str:
    tokens = tokenizer.tokenize(norm_tweet.replace("’", "'").replace("…", "..."))
    norm_tweet = " ".join([normalize_token(token) for token in tokens])

    norm_tweet = norm_tweet.replace("n't ", " n't ").replace("n 't ", " n't ").replace("n ' t ", " n't ")
    norm_tweet = norm_tweet.replace("cannot ", "can not ").replace("ca n't", "can't")
    norm_tweet = norm_tweet.replace("wo n't", "won't").replace("ai n't", "ain't")
    norm_tweet = norm_tweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ")
    norm_tweet = norm_tweet.replace("'ll ", " 'll ").replace("'d ", " 'd ").replace("'ve ", " 've ")
    norm_tweet = norm_tweet.replace(" p . m .", "  p.m.").replace(" p . m ", " p.m ")
    norm_tweet = norm_tweet.replace(" a . m .", " a.m.").replace(" a . m ", " a.m ")

    norm_tweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", norm_tweet)
    norm_tweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", norm_tweet)
    norm_tweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", norm_tweet)

    if to_ascii:
        norm_tweet = unicodedata.normalize('NFKD', norm_tweet).encode('ascii', 'ignore').decode('utf-8')
    if to_lower:
        norm_tweet = norm_tweet.lower()
    return " ".join(norm_tweet.split())


def normalize_series(tweet_series: pd.Series, to_ascii=True) -> pd.Series:
    return tweet_series.apply(lambda txt: normalize_text(txt, to_ascii=to_ascii))


if __name__ == "__main__":
    print(normalize_text(
        "SC has first two presumptive cases of coronavirus, DHEC confirms "
        "https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms"
        "/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user"
        "-share… via @postandcourier"))

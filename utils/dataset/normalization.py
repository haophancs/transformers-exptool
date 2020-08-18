from nltk.tokenize import TweetTokenizer
from emoji import demojize, emoji_count
import pandas as pd
import re
import unicodedata
import preprocessor as p
from ekphrasis.classes.segmenter import Segmenter
from pycontractions import Contractions

cont = Contractions(api_key="glove-twitter-100")
cont.load_models()
seg_tw = Segmenter(corpus="twitter")
w_tokenizer = TweetTokenizer()


def handle_special_characters(norm_tweet):
    norm_tweet = re.sub(r"\x89Û_", "", norm_tweet)
    norm_tweet = re.sub(r"\x89ÛÒ", "", norm_tweet)
    norm_tweet = re.sub(r"\x89ÛÓ", "", norm_tweet)
    norm_tweet = re.sub(r"\x89ÛÏWhen", "When", norm_tweet)
    norm_tweet = re.sub(r"\x89ÛÏ", "", norm_tweet)
    norm_tweet = re.sub(r"China\x89Ûªs", "China's", norm_tweet)
    norm_tweet = re.sub(r"let\x89Ûªs", "let's", norm_tweet)
    norm_tweet = re.sub(r"\x89Û÷", "", norm_tweet)
    norm_tweet = re.sub(r"\x89Ûª", "", norm_tweet)
    norm_tweet = re.sub(r"\x89Û\x9d", "", norm_tweet)
    norm_tweet = re.sub(r"å_", "", norm_tweet)
    norm_tweet = re.sub(r"\x89Û¢", "", norm_tweet)
    norm_tweet = re.sub(r"\x89Û¢åÊ", "", norm_tweet)
    norm_tweet = re.sub(r"fromåÊwounds", "from wounds", norm_tweet)
    norm_tweet = re.sub(r"åÊ", "", norm_tweet)
    norm_tweet = re.sub(r"åÈ", "", norm_tweet)
    norm_tweet = re.sub(r"JapÌ_n", "Japan", norm_tweet)
    norm_tweet = re.sub(r"Ì©", "e", norm_tweet)
    norm_tweet = re.sub(r"å¨", "", norm_tweet)
    norm_tweet = re.sub(r"SuruÌ¤", "Suruc", norm_tweet)
    norm_tweet = re.sub(r"åÇ", "", norm_tweet)
    norm_tweet = re.sub(r"å£3million", "3 million", norm_tweet)
    norm_tweet = re.sub(r"åÀ", "", norm_tweet)
    norm_tweet = re.sub("’", "'", norm_tweet)
    norm_tweet = re.sub("…", "...", norm_tweet)
    norm_tweet = re.sub("ー", "-", norm_tweet)
    norm_tweet = re.sub(r"&gt;", ">", norm_tweet)
    norm_tweet = re.sub(r"&lt;", "<", norm_tweet)
    norm_tweet = re.sub(r"&amp;", "&", norm_tweet)
    return norm_tweet


def handle_contractions(norm_tweet):
    # Contractions
    norm_tweet = re.sub(r"don\x89Ûªt", "do not", norm_tweet)
    norm_tweet = re.sub(r"I\x89Ûªm", "I am", norm_tweet)
    norm_tweet = re.sub(r"you\x89Ûªve", "you have", norm_tweet)
    norm_tweet = re.sub(r"it\x89Ûªs", "it is", norm_tweet)
    norm_tweet = re.sub(r"doesn\x89Ûªt", "does not", norm_tweet)
    norm_tweet = re.sub(r"It\x89Ûªs", "It is", norm_tweet)
    norm_tweet = re.sub(r"Here\x89Ûªs", "Here is", norm_tweet)
    norm_tweet = re.sub(r"I\x89Ûªve", "I have", norm_tweet)
    norm_tweet = re.sub(r"can\x89Ûªt", "cannot", norm_tweet)
    norm_tweet = re.sub(r"That\x89Ûªs", "That is", norm_tweet)
    norm_tweet = re.sub(r"You\x89Ûªre", "You are", norm_tweet)
    norm_tweet = re.sub(r"Don\x89Ûªt", "Do not", norm_tweet)
    norm_tweet = re.sub(r"Can\x89Ûªt", "Cannot", norm_tweet)
    norm_tweet = re.sub(r"you\x89Ûªll", "you will", norm_tweet)
    norm_tweet = re.sub(r"I\x89Ûªd", "I would", norm_tweet)
    norm_tweet = re.sub(r"donå«t", "do not", norm_tweet)
    norm_tweet = list(cont.expand_texts([norm_tweet]))[0]
    return norm_tweet


def handle_abbreviations(norm_tweet):
    norm_tweet = re.sub(r"U\.S", "United States", norm_tweet)
    norm_tweet = re.sub(r"w/e", "whatever", norm_tweet)
    norm_tweet = re.sub(r"w/", "with", norm_tweet)
    norm_tweet = re.sub(r"USAgov", "USA government", norm_tweet)
    norm_tweet = re.sub(r"recentlu", "recently", norm_tweet)
    norm_tweet = re.sub(r"Ph0tos", "Photos", norm_tweet)
    norm_tweet = re.sub(r"amirite", "am I right", norm_tweet)
    norm_tweet = re.sub(r"exp0sed", "exposed", norm_tweet)
    norm_tweet = re.sub(r"<3", "love", norm_tweet)
    norm_tweet = re.sub(r"amageddon", "armageddon", norm_tweet)
    norm_tweet = re.sub(r"Trfc", "Traffic", norm_tweet)
    norm_tweet = re.sub(r"16yr", "16 years", norm_tweet)
    norm_tweet = re.sub(r"lmao", "laughing my ass off", norm_tweet)
    norm_tweet = re.sub(r"TRAUMATISED", "traumatized", norm_tweet)
    norm_tweet = norm_tweet.replace("cv19", "COVID19")
    norm_tweet = norm_tweet.replace("cvid19", "COVID19")
    return norm_tweet


def handle_hashtag(norm_tweet):
    for hashtag in re.findall(r"#(\w+)", norm_tweet):
        norm_tweet = norm_tweet.replace(f'#{hashtag}', '#'+seg_tw.segment(hashtag))
    return norm_tweet


def normalize_token(token, keep_emojis=True, username="@USER", httpurl="httpurl"):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return username
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return httpurl
    elif len(token) == 1:
        if keep_emojis:
            demojized = demojize(token)
            if ":regional_indicator_symbol_letter_" in demojized:
                return ""
            if ":globe" in demojized:
                return ":globe:"
            return demojized
        if emoji_count(token) > 0:
            return ""
        return token
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        elif token == "ー":
            return "-"
        else:
            return token


def normalize_text(norm_tweet, to_ascii=True, to_lower=False, keep_emojis=True, username="@USER",
                   httpurl="httpurl") -> str:
    norm_tweet = re.sub(r"(covid.19)", "COVID19 ", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r"covid19", " COVID19 ", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r"# COVID19", "#COVID19", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r'\s+', ' ', norm_tweet).strip()
    norm_tweet = handle_special_characters(norm_tweet)
    norm_tweet = handle_contractions(norm_tweet)
    norm_tweet = handle_abbreviations(norm_tweet)
    norm_tweet = handle_hashtag(norm_tweet)
    p.set_options(p.OPT.RESERVED, p.OPT.SMILEY)
    norm_tweet = p.clean(norm_tweet)

    tokens = w_tokenizer.tokenize(norm_tweet.replace("’", "'").replace("…", "..."))
    norm_tweet = " ".join([normalize_token(token,
                                           keep_emojis=keep_emojis,
                                           username=username,
                                           httpurl=httpurl) for token in tokens])
    if '...' not in norm_tweet:
        norm_tweet = norm_tweet.replace('..', ' ... ')

    norm_tweet = norm_tweet.replace("n't ", " n't ").replace("n 't ", " n't ").replace("n ' t ", " n't ")
    norm_tweet = norm_tweet.replace("cannot ", "can not ").replace("ca n't", "can not")
    norm_tweet = norm_tweet.replace("ai n't", "ain't").replace(" n't", " not")
    norm_tweet = norm_tweet.replace("'m ", " am ").replace("'re ", " are ").replace("'s ", " 's ")
    norm_tweet = norm_tweet.replace("'ll ", " will ").replace("'d ", " would ").replace("'ve ", " have ")
    norm_tweet = norm_tweet.replace(" p . m .", "  p.m.").replace(" p . m ", " p.m ")
    norm_tweet = norm_tweet.replace(" a . m .", " a.m.").replace(" a . m ", " a.m ")
    # norm_tweet = norm_tweet.replace('\"\"', ' ')

    norm_tweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", norm_tweet)
    norm_tweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", norm_tweet)
    norm_tweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", norm_tweet)

    norm_tweet = re.sub(r"(covid.19)", "COVID19", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r'\s+', ' ', norm_tweet).strip()

    if to_ascii:
        norm_tweet = unicodedata.normalize('NFKD', norm_tweet).encode('ascii', 'ignore').decode('utf-8')
    if to_lower:
        norm_tweet = norm_tweet.lower()
    return norm_tweet


def normalize_series(tweet_series: pd.Series,
                     to_ascii=True,
                     to_lower=False,
                     keep_emojis=True,
                     username="@USER",
                     httpurl="HTTPURL") -> pd.Series:
    return tweet_series.apply(lambda txt: normalize_text(txt,
                                                         to_ascii=to_ascii,
                                                         to_lower=to_lower,
                                                         keep_emojis=keep_emojis,
                                                         username=username,
                                                         httpurl=httpurl))


if __name__ == "__main__":
    print(normalize_text(
        "SC has first two presumptive cases of coronavirus, DHEC confirms "
        "https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms"
        "/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user"
        "-share… via @postandcourier"))

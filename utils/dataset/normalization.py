from nltk.tokenize import TweetTokenizer
from emoji import demojize, emoji_count
import pandas as pd
import re
import html
import unicodedata
import unidecode
import preprocessor as p
from ekphrasis.classes.segmenter import Segmenter
from pycontractions import Contractions

cont = Contractions(api_key="glove-twitter-100")
cont.load_models()
seg_tw = Segmenter(corpus="twitter")
w_tokenizer = TweetTokenizer()
control_char_regex = re.compile(r'[\r\n\t]+')
transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–ー-", u"'''\"\"---")])


def normalize_punctuation(norm_tweet):
    # handle punctuation
    norm_tweet = norm_tweet.translate(transl_table)
    norm_tweet = norm_tweet.replace('…', '...')
    norm_tweet = ''.join([unidecode.unidecode(t) if unicodedata.category(t)[0] == 'P' else t for t in norm_tweet])
    if '...' not in norm_tweet:
        norm_tweet = norm_tweet.replace('..', ' ... ')
    return norm_tweet


def normalize_special_characters(norm_tweet):
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
    norm_tweet = html.unescape(norm_tweet)
    return norm_tweet


def normalize_contractions(norm_tweet):
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
    norm_tweet = re.sub(r"that\x89Ûªs", "that is", norm_tweet)
    norm_tweet = re.sub(r"This\x89Ûªs", "This is", norm_tweet)
    norm_tweet = re.sub(r"this\x89Ûªs", "this is", norm_tweet)
    norm_tweet = re.sub(r"You\x89Ûªre", "You are", norm_tweet)
    norm_tweet = re.sub(r"Don\x89Ûªt", "Do not", norm_tweet)
    norm_tweet = re.sub(r"Can\x89Ûªt", "Cannot", norm_tweet)
    norm_tweet = re.sub(r"you\x89Ûªll", "you will", norm_tweet)
    norm_tweet = re.sub(r"I\x89Ûªd", "I would", norm_tweet)
    norm_tweet = re.sub(r"donå«t", "do not", norm_tweet)

    norm_tweet = re.sub(r"He's", "He is", norm_tweet)
    norm_tweet = re.sub(r"She's", "She is", norm_tweet)
    norm_tweet = re.sub(r"It's", "It is", norm_tweet)
    norm_tweet = re.sub(r"he's", "he is", norm_tweet)
    norm_tweet = re.sub(r"she's", "she is", norm_tweet)
    norm_tweet = re.sub(r"it's", "it is", norm_tweet)

    norm_tweet = re.sub(r"He ain't", "He is not", norm_tweet)
    norm_tweet = re.sub(r"She aint't", "She is not", norm_tweet)
    norm_tweet = re.sub(r"It aint't", "It is not", norm_tweet)
    norm_tweet = re.sub(r"he aint't", "he is not", norm_tweet)
    norm_tweet = re.sub(r"she aint't", "she is not", norm_tweet)
    norm_tweet = re.sub(r"it aint't", "it is not", norm_tweet)
    norm_tweet = list(cont.expand_texts([norm_tweet]))[0]
    return norm_tweet


def normalize_abbreviations(norm_tweet):
    norm_tweet = re.sub(r'R\.I\.P', 'Rest In Peace', norm_tweet)
    norm_tweet = re.sub(r'R\.i\.p', 'Rest in peace', norm_tweet)
    norm_tweet = re.sub(r'r\.i\.p', 'rest in peace', norm_tweet)
    norm_tweet = re.sub(r"U\.S", "United States", norm_tweet)
    norm_tweet = re.sub(r"u\.s", "united states", norm_tweet)
    norm_tweet = re.sub(r"w/e", "whatever", norm_tweet)
    norm_tweet = re.sub(r"w/", "with", norm_tweet)
    norm_tweet = re.sub(r"USAgov", "USA government", norm_tweet)
    norm_tweet = re.sub(r"usagov", "usa government", norm_tweet)
    norm_tweet = re.sub(r"recentlu", "recently", norm_tweet)
    norm_tweet = re.sub(r"Ph0tos", "Photos", norm_tweet)
    norm_tweet = re.sub(r"ph0tos", "photos", norm_tweet)
    norm_tweet = re.sub(r"amirite", "am I right", norm_tweet)
    norm_tweet = re.sub(r"exp0sed", "exposed", norm_tweet)
    norm_tweet = re.sub(r"<3", "love", norm_tweet)
    norm_tweet = re.sub(r"amageddon", "armageddon", norm_tweet)
    norm_tweet = re.sub(r"Trfc", "Traffic", norm_tweet)
    norm_tweet = re.sub(r"trfc", "traffic", norm_tweet)
    norm_tweet = re.sub(r"([0-9]+)(yr)", r"\1 years", norm_tweet)
    norm_tweet = re.sub(r"lmao", "laughing my ass off", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r"lol", "laughing out loud", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r"TRAUMATISED", "traumatized", norm_tweet)
    norm_tweet = re.sub(r"traumatised", "traumatized", norm_tweet)
    norm_tweet = re.sub(r"ppl", "people", norm_tweet)
    norm_tweet = re.sub(r"Ppl", "People", norm_tweet)
    norm_tweet = re.sub(r"sh\*t", r"shit", norm_tweet)
    norm_tweet = norm_tweet.replace("cv19", "COVID 19")
    norm_tweet = norm_tweet.replace("cvid19", "COVID 19")
    return norm_tweet


def normalize_hashtag(norm_tweet):
    for hashtag in re.findall(r"#(\w+)", norm_tweet):
        norm_tweet = norm_tweet.replace(f'#{hashtag}', seg_tw.segment(hashtag))
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
        elif emoji_count(token) > 0:
            return ""
    return token


def replace_multi_occurrences(norm_tweet, filler):
    # only run if we have multiple occurrences of filler
    if norm_tweet.count(filler) <= 1:
        return norm_tweet
    # pad fillers with whitespace
    norm_tweet = norm_tweet.replace(f'{filler}', f' {filler} ')
    # remove introduced duplicate whitespaces
    norm_tweet = ' '.join(norm_tweet.split())
    # find indices of occurrences
    indices = []
    for m in re.finditer(r'{}'.format(filler), norm_tweet):
        index = m.start()
        indices.append(index)
    # collect merge list
    merge_list = []
    old_index = None
    for i, index in enumerate(indices):
        if i > 0 and index - old_index == len(filler) + 1:
            # found two consecutive fillers
            if len(merge_list) > 0 and merge_list[-1][1] == old_index:
                # extend previous item
                merge_list[-1][1] = index
                merge_list[-1][2] += 1
            else:
                # create new item
                merge_list.append([old_index, index, 2])
        old_index = index
    # merge occurrences
    if len(merge_list) > 0:
        new_text = ''
        pos = 0
        for (start, end, count) in merge_list:
            new_text += norm_tweet[pos:start]
            new_text += f'{count} {filler}'
            pos = end + len(filler)
        new_text += norm_tweet[pos:]
        norm_tweet = new_text
    return norm_tweet


def normalize_text(norm_tweet,
                   to_ascii=True,
                   to_lower=False,
                   keep_emojis=True,
                   segment_hashtag=True,
                   username="@USER",
                   httpurl="httpurl") -> str:
    if to_lower:
        norm_tweet = norm_tweet.lower()

    norm_tweet = normalize_special_characters(norm_tweet)
    norm_tweet = normalize_punctuation(norm_tweet)
    norm_tweet = normalize_contractions(norm_tweet)
    norm_tweet = normalize_abbreviations(norm_tweet)

    tokens = w_tokenizer.tokenize(norm_tweet)
    norm_tweet = " ".join([normalize_token(token,
                                           keep_emojis=keep_emojis,
                                           username=username,
                                           httpurl=httpurl) for token in tokens])
    norm_tweet = replace_multi_occurrences(norm_tweet, username)
    norm_tweet = replace_multi_occurrences(norm_tweet, httpurl)

    norm_tweet = re.sub(r"(covid.19)", "COVID 19 ", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r"(covid...19)", "COVID 19 ", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r"covid19", " COVID 19 ", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r"# COVID19", "#COVID 19", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r"# COVID19", "#COVID 19", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r'\s+', ' ', norm_tweet).strip()

    if segment_hashtag:
        norm_tweet = normalize_hashtag(norm_tweet)
    p.set_options(p.OPT.RESERVED, p.OPT.SMILEY)
    norm_tweet = p.clean(norm_tweet)

    # replace \t, \n and \r characters by a whitespace
    norm_tweet = re.sub(control_char_regex, ' ', norm_tweet)
    # remove all remaining control characters
    norm_tweet = ''.join(ch for ch in norm_tweet if unicodedata.category(ch)[0] != 'C')

    norm_tweet = re.sub(r" p \. m \.", "  p.m.", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r" p \. m ", " p.m ", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r" a \. m \.", "  a.m.", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r" a \. m ", " a.m ", norm_tweet, flags=re.I)
    norm_tweet = re.sub(r"'s", " 's ", norm_tweet)
    norm_tweet = re.sub(r"(covid.19)", "COVID19", norm_tweet, flags=re.I)

    norm_tweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", norm_tweet)
    norm_tweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", norm_tweet)
    norm_tweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", norm_tweet)

    if to_ascii:
        norm_tweet = ''.join(ch for ch in norm_tweet if unicodedata.category(ch)[0] != 'So')
        norm_tweet = unicodedata.normalize('NFKD', norm_tweet).encode('ascii', 'ignore').decode('utf-8')
    if to_lower:
        norm_tweet = norm_tweet.lower()

    while '""' in norm_tweet:
        norm_tweet = norm_tweet.replace('""', '"')
    norm_tweet = re.sub(r'\"+', '"', norm_tweet)
    norm_tweet = re.sub(r'\s+', ' ', norm_tweet).strip()
    return norm_tweet


def normalize_series(tweet_series: pd.Series,
                     to_ascii=True,
                     to_lower=False,
                     keep_emojis=True,
                     segment_hashtag=False,
                     username="@USER",
                     httpurl="HTTPURL") -> pd.Series:
    return tweet_series.apply(lambda txt: normalize_text(txt,
                                                         to_ascii=to_ascii,
                                                         to_lower=to_lower,
                                                         keep_emojis=keep_emojis,
                                                         segment_hashtag=segment_hashtag,
                                                         username=username,
                                                         httpurl=httpurl))


if __name__ == "__main__":
    print(normalize_text(
        "SC has first two presumptive cases of coronavirus, DHEC confirms "
        "https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms"
        "/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user"
        "-share… via @postandcourier #Covid_19"))

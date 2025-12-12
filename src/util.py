# Util file for quickly returning dataframe of cleaned data

from pathlib import Path
import pandas as pd
import numpy as np
import math
import re
from urllib.parse import urlparse, urlsplit
from collections import Counter
import string

# for copy on write
pd.set_option("mode.copy_on_write", True)

from sklearn.feature_extraction.text import TfidfVectorizer


#cwd = Path().cwd()
#project_folder = cwd.parent
project_folder = Path(__file__).resolve().parents[1]

KEYWORDS = [
    "pay", "ticket", "fine", "refund", "claim", "submit", "verify", "update",
    "account", "login", "secure", "reset", "password", "invoice", "shipping",
    "delivery", "package"
]


keyword_pattern = re.compile("|".join(KEYWORDS), re.IGNORECASE)


#Assuming you have original datasets

def generate_clean_csv():

    data_folder = project_folder / "csv"

    df1 = pd.read_csv(data_folder / "Phishing URLs.csv")
    print("df1 shape: ", df1.shape)
    df1 = df1.rename(columns={"Type": "type"})[['url', 'type']]

    df2 = pd.read_csv(data_folder /"urldata.csv")
    print("df2 shape: ", df2.shape)
    ##Tests
    print("df2columns", df2.columns)
    # normalize urldata.csv
    if "label" in df2.columns:
        df2 = df2.rename(columns={"label": "type"})
    df2 = df2[["url", "type"]]


    df3 = pd.read_csv(data_folder /"malicious_phish.csv")
    print("df3 shape: ", df3.shape)

    final = pd.concat([df1, df2, df3], ignore_index=True)
    final = final.dropna(subset=['url', 'type'])
    final = final.drop_duplicates(subset=['url']).reset_index(drop=True)

    final = final.drop(columns=[c for c in ["Unnamed: 0", "result"] if c in final.columns])

    final["type"] = final["type"].astype(str).str.strip().str.lower()


    label_map = {
        "legitimate": 0,
        "benign": 0,
        "safe": 0,  # in case this appears
        "phishing": 1,
        "malware": 1,
        "defacement": 1
    }

    final["label"] = final["type"].map(label_map)

    # Drop non classed labels
    final = final.dropna(subset=["label"]).reset_index(drop=True)

    # convert to int
    final["label"] = final["label"].astype(int)

    final['Valid_URLs'] = final['url'].apply(is_reasonable_url)

    final = final[final['Valid_URLs']]

    final_cleaned  = final.drop(columns=['Valid_URLs'])
    # final_cleaned = final.dropna(subset=['Valid_URLs'])
    # final_cleaned["url"] = final_cleaned["Valid_URLs"]

    # final_cleaned = final_cleaned.drop(columns=['Valid_URLs'])

    final_cleaned.to_csv(data_folder / "all_urls.csv", index=False)

def is_reasonable_url(u):
    p = urlparse(u)
    return bool(p.scheme or p.netloc or p.path)

def is_valid_url(url):

    try:
        parsed_url = urlparse(str(url))
        if parsed_url.scheme and parsed_url.netloc:
            return url
        else:
            return np.nan
    except ValueError:
        return np.nan


def safe_div(num, den):
    return (num / den) if den else 0.0


def char_proportions(url: str):
    if not url:
        return 0.0, 0.0, 0.0

    total = len(url)
    if total == 0:
        return 0.0, 0.0, 0.0

    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    specials = sum(c in string.punctuation for c in url)

    return (
        digits / total,
        letters / total,
        specials / total
    )



def keyword_count(url: str):
    matches = keyword_pattern.findall(url)
    return len(matches)



#char class
# 1 = alpha, 2 = digit, 3 = special 4 = other
def get_char_class(c: str) -> int:
    if c.isalpha():
        return 1
    if c.isdigit():
        return 2
    if c in "/?&=.-_~%+@:#":
        return 3
    return 4



def get_runs(seq) -> int:
    if not seq:
        return 0
    r = 1
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            r += 1
    return r



#CCR
# Character Continuity Rate is used to find the sum
# of the longest token length of each character type in the domain, such as
# abc567ti = (3 + 3 + 1)/9 = 0.77.
# measures how often the url transitions between different character classes
# closer to 1 typically indicates valid url
# closer to 0 indicates invalid url
def get_CCR_transition(url:str) -> float:

    s = (url or "").strip()

    if len(s) <= 1:
        return 0.0
    else:
        seq = [get_char_class(c) for c in s]
        r = get_runs(seq)
        return 1.0 - safe_div((r - 1), (len(s) - 1))

# gets the average length of each token in the path token
def average_path_token_length(url: str) -> float:
    try:
        parts = urlsplit(url if "://" in url else "http://" + url)
        path = parts.path or ""
        tokens = [t for t in path.split("/") if t]

        if not tokens:
            return 0.0

        return sum(len(t) for t in tokens) / len(tokens)

    except ValueError:
        # Invalid IPv6 or malformed URL
        return 0.0

# Shannon entropy function
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())



# Extract core lexical features
def extract_lexical_features(url: str):
    #print("current url: ", url )
    parsed = urlparse(url)

    host = parsed.netloc
    path = parsed.path + ("?" + parsed.query if parsed.query else "")

    # lengths
    url_len = len(url)
    host_len = len(host)
    path_len = len(path)


    # Char counts
    num_digits = sum(c.isdigit() for c in url)
    num_special = sum(c in string.punctuation for c in url)
    num_letters = sum(c.isalpha() for c in url)


    #URL structure counts
    num_dots = url.count('.')
    num_slashes = url.count('/')

    # shannon entropy for randomnes
    entropy_value = shannon_entropy(url)
    #keyword count
    k_count = keyword_count(url)
    ccr_transition = get_CCR_transition(url)
    avg_path_tok_len = average_path_token_length(url)
    digit_prop, letter_prop, special_prop = char_proportions(url)

    return pd.Series({
        "url_length": url_len,
        "host_length": host_len,
        "path_length": path_len,
        "num_digits": num_digits,
        "num_special": num_special,
        "num_letters": num_letters,
        "num_dots": num_dots,
        "num_slashes": num_slashes,
        "entropy": entropy_value,
        "keyword_count": k_count,
        "digit_proportion": digit_prop,
        "letter_proportion": letter_prop,
        "special_proportion": special_prop,
        "ccr_transition": ccr_transition,
        "avg_path_token_length": avg_path_tok_len
    })


def get_lexical_features(df):
    lexical_features = df["url"].apply(extract_lexical_features)
    df = pd.concat([df, lexical_features], axis=1)

    return df


def generate_final_dataset():
    df = pd.read_csv(project_folder / "csv" / "all_urls.csv")
    print(df.head())
    print(df.shape)
    df = get_lexical_features(df)
    df.to_csv(project_folder / "csv" / "final_dataset.csv", index=False)

def test_get_runs():

    assert get_runs([]) == 0

    assert get_runs(["a"]) == 1

    assert get_runs(["a", "a", "a"]) == 1

    assert get_runs(["a", "b", "c", "d"]) == 4

    assert get_runs(["a", "b", "a", "b"]) == 4

    assert get_runs(["a", "a", "b", "b", "a"]) == 3

    assert get_runs([1, 1, 2, 2, 2, 3]) == 3
    print("All tests passed!")


def test_getCCR():
    # Empty and single
    assert get_CCR_transition("") == 0.0
    assert get_CCR_transition("a") == 0.0

    assert abs(get_CCR_transition("aaaa") - 1.0) < 1e-12
    assert abs(get_CCR_transition("1111") - 1.0) < 1e-12

    assert abs(get_CCR_transition("a1a1") - 0.0) < 1e-12


    assert abs(get_CCR_transition("aaa111") - 0.8) < 1e-12


    assert abs(get_CCR_transition("ab--12") - 0.6) < 1e-12


    print(get_CCR_transition(""))
    print(get_CCR_transition("a"))
    print(get_CCR_transition("aaaa"))
    print(get_CCR_transition("1111"))
    print(get_CCR_transition("a1a1"))
    print(get_CCR_transition("aaa111"))
    print(get_CCR_transition("ab--12"))






def test_average_path_token_length():
    # no path
    assert average_path_token_length("http://example.com") == 0.0

    # one path var length = 5
    assert average_path_token_length("http://example.com/login") == 5.0

    # two path len1 = 8, len2 = 5 = 13/2 = 6.5
    assert abs(
        average_path_token_length("http://example.com/products/books") - 6.5
    ) < 1e-12

    # no path var = 0
    assert average_path_token_length("http://example.com/") == 0.0

    # three path len1 = 1 , len2 = does not count in this algo , len3 = 2, len4 = 1 = 4/3 = 1.33
    assert abs(
        average_path_token_length("http://example.com/a//bc/d") - (4 / 3)
    ) < 1e-12



if __name__ == "__main__": #change after tests
    test_get_runs()
    test_getCCR()
    test_average_path_token_length()
    generate_clean_csv()
    generate_final_dataset()
    
    csv_path = project_folder / "csv" / "all_urls.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(df.shape)
        print(df.isna().sum().sum())
    else:
        print("all_urls.csv not found")

    """
    My output after running:
    All tests passed!
    0.0
    0.0
    1.0
    1.0
    0.0
    0.8
    0.6
    (695859, 3)
    0
    """
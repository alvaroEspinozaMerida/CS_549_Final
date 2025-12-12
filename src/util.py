# Util file for quickly returning dataframe of cleaned data

from pathlib import Path
import pandas as pd
import numpy as np
import math
import re
from urllib.parse import urlparse
from collections import Counter
import string

# for copy on write
pd.set_option("mode.copy_on_write", True)

from sklearn.feature_extraction.text import TfidfVectorizer


cwd = Path().cwd()
project_folder = cwd.parent

KEYWORDS = [
    "pay", "ticket", "fine", "refund", "claim", "submit", "verify", "update",
    "account", "login", "secure", "reset", "password", "invoice", "shipping",
    "delivery", "package"
]


keyword_pattern = re.compile("|".join(KEYWORDS), re.IGNORECASE)


#Assuming you have original datasets

def generate_clean_csv():

    data_folder = project_folder / "data"

    df1 = pd.read_csv(data_folder / "Phishing URLs.csv")
    print("df1 shape: ", df1.shape)
    df1 = df1.rename(columns={"Type": "type"})[['url', 'type']]

    df2 = pd.read_csv(data_folder /"urldata.csv")
    print("df2 shape: ", df2.shape)

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

    final_cleaned.to_csv(data_folder / "all_urls.csv")

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



def keyword_count(url: str):
    matches = keyword_pattern.findall(url)
    return len(matches)




#CCR
# Character Continuity Rate is used to find the sum
# of the longest token length of each character type in the domain, such as
# abc567ti = (3 + 3 + 1)/9 = 0.77.

# Shannon entropy function
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())



# Extract core lexical features
def extract_lexical_features(url: str):
    print("current url: ", url )
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
        "keyword_count": k_count
    })


def get_lexical_features(df):
    lexical_features = df["url"].apply(extract_lexical_features)
    df = pd.concat([df, lexical_features], axis=1)

    return df


def generate_final_dataset():
    df = pd.read_csv(project_folder / "data" / "all_urls.csv")
    print(df.head())
    print(df.shape)
    df = get_lexical_features(df)
    df.to_csv(project_folder / "data" / "final_dataset.csv", index=False)

# generate_final_dataset()
# generate_clean_csv()
df = pd.read_csv(project_folder / "data" / "all_urls.csv")
print(df.head())
print(df.shape)
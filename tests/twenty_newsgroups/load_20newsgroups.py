import logging
import pickle
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd

'''
Mostly taken from: https://github.com/cemoody/lda2vec/blob/master/examples/twenty_newsgroups/data/preprocess.py
'''
# Fetch data
remove = ('headers', 'footers', 'quotes')
texts = fetch_20newsgroups(subset='train', remove=remove).data
# Remove tokens with these substrings
bad = set(["ax>", '`@("', '---', '===', '^^^'])

def clean(line):
    return ' '.join(w for w in line.split() if not any(t in w for t in bad))

# Preprocess data
max_length = 10000   # Limit of 10k words per document
# Convert to unicode (spaCy only works with unicode)
texts = [str(clean(d)) for d in texts]

# Convert text to dataframe
df = pd.DataFrame(texts)
df.to_csv("20_newsgroups.txt", sep="\t", index=False, header=None)

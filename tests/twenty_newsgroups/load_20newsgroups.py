import pickle
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd
from lda2vec.nlppipe import NlpPipeline
from keras.preprocessing.sequence import skipgrams
import os
import time

'''
Partially stolen from: https://github.com/cemoody/lda2vec/blob/master/examples/twenty_newsgroups/data/preprocess.py
'''
# Path to text file
path_to_file = "20_newsgroups.txt"
# Location where data will be stored with the given hyperparameters
run_name = "skipgrams_clean_full"
# Name of the file that the skipgram pairs will be saved to
file_out_path = run_name + "/" + path_to_file[:-4] + "_" + run_name + ".txt"
# Set minimum frequency threshold
min_freq_threshold = 10


def clean(line):
    return ' '.join(w for w in line.split() if not any(t in w for t in bad))


start = time.time()

if not os.path.exists(file_out_path):

    # Make directory to save data in
    os.makedirs(run_name)
    # Fetch data
    remove = ('headers', 'footers', 'quotes')
    texts = fetch_20newsgroups(subset='train', remove=remove).data
    # Remove tokens with these substrings
    bad = set(["ax>", '`@("', '---', '===', '^^^', "AX>", "GIZ"])

    # Preprocess data
    max_length = 10000  # Limit of 10k words per document

    # Convert to unicode (spaCy only works with unicode)
    texts = [str(clean(d)) for d in texts]

    # Process the text, no file because we are passing in data directly
    SP = NlpPipeline(None, max_length, texts=texts,
                     num_threads=10, only_keep_alpha=True,
                     token_type="lemma", vectors="google_news_model")

    # Computes the embed matrix along with other variables
    SP._compute_embed_matrix()

    print("converting data to w2v indexes")
    # Convert data to word2vec indexes
    SP.convert_data_to_word2vec_indexes()

    print("trimming 0's")
    # Trim zeros from idx data
    SP.trim_zeros_from_idx_data()

    # This extracts the length of each document (needed for pyldaviz)
    doc_lengths = [len(x) for x in SP.idx_data]

    # Find the cutoff idx
    for i, freq in enumerate(SP.freqs):
        if freq < min_freq_threshold:
            cutoff = i
            break
    # Then, cut off the embed matrix
    embed_matrix = SP.embed_matrix[:cutoff]
    # Also, replace all tokens below cutoff in idx_data
    for i in range(len(SP.idx_data)):
        SP.idx_data[i][SP.idx_data[i] > cutoff - 1] = 0
    # Next, cut off the frequencies
    freqs = SP.freqs[:cutoff]

    print("converting to skipgrams")
    write_every = 10000
    data = []
    num_examples = SP.idx_data.shape[0]
    # Sometimes docs can be less than the required amount for
    # the skipgram function. So, we must manually make a counter
    # instead of relying on the enumerated index (i)
    doc_id_counter = 0
    # Additionally, we will keep track of these lower level docs
    # and will purge them later
    purged_docs = []
    for i, t in enumerate(SP.idx_data):
        pairs, _ = skipgrams(t,
                             vocabulary_size=SP.vocab_size,
                             window_size=5,
                             shuffle=True,
                             negative_samples=0)
        # Pairs will be 0 if document is less than 2 indexes
        if len(pairs) > 2:
            for pair in pairs:
                temp_data = pair
                # Appends doc ID
                temp_data.append(doc_id_counter)
                # Appends document index
                temp_data.append(i)
                data.append(temp_data)
            doc_id_counter += 1
        else:
            purged_docs.append(i)
        if i // write_every:
            temp_df = pd.DataFrame(data)
            temp_df.to_csv(file_out_path, sep="\t", index=False, header=None, mode="a")
            del temp_df
            data = []
        if i % 500 == 0:
            print("step", i, "of", num_examples)
    temp_df = pd.DataFrame(data)
    temp_df.to_csv(file_out_path, sep="\t", index=False, header=None, mode="a")
    del temp_df

    # Save embed matrix
    np.save(run_name + "/embed_matrix", embed_matrix)
    # Save the doc lengths to be used later, also, purge those that didnt make it into skipgram function
    np.save(run_name + "/doc_lengths", np.delete(doc_lengths, np.array(purged_docs)))
    # Save frequencies to file
    np.save(run_name + "/freqs", freqs)
    # Save vocabulary dictionaries to file
    idx_to_word_out = open(run_name + "/" + "idx_to_word.pickle", "wb")
    pickle.dump(SP.idx_to_word, idx_to_word_out)
    idx_to_word_out.close()
    word_to_idx_out = open(run_name + "/" + "word_to_idx.pickle", "wb")
    pickle.dump(SP.word_to_idx, word_to_idx_out)
    word_to_idx_out.close()

print("The whole program took", time.time() - start, "seconds")

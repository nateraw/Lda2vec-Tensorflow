from lda2vec import model
import numpy as np
import pandas as pd
import tensorflow as tf
#from nlppipeline.nlppipe import NlpPipeline
from lda2vec.nlppipe import NlpPipeline
from keras.preprocessing.sequence import skipgrams
import os
from sklearn.utils import shuffle
import pickle

path_to_file  = "20_newsgroups.txt"
file_out_path = path_to_file[:-4] + "_skipgrams.txt"


# If the file_out_path has not been created, it will be here
if not os.path.isfile(file_out_path):
    SP = NlpPipeline(path_to_file, 50, merge=True, num_threads=8, context=True, usecols=["texts"], vectors="google_news_model")
    # Compute the embedding matrix
    SP._compute_embed_matrix(random=True)
    # Convert data to word2vec indexes
    SP.convert_data_to_word2vec_indexes()
    # Trim zeros from idx data
    SP.trim_zeros_from_idx_data()

    # Save frequencies to file
    np.save("freqs", SP.freqs)

    # Save vocabulary dictionaries to file
    idx_to_word_out = open("idx_to_word.pickle", "wb")
    pickle.dump(SP.idx_to_word, idx_to_word_out)
    idx_to_word_out.close()
    word_to_idx_out = open("word_to_idx.pickle", "wb")
    pickle.dump(SP.word_to_idx, word_to_idx_out)
    word_to_idx_out.close()

    write_every = 10000
    data = []
    for i, t in enumerate(SP.idx_data):
        pairs, _ = skipgrams(t,
                 vocabulary_size = SP.vocab_size,
                 window_size = 5,
                 shuffle=True,
                 negative_samples=0)
        for pair in pairs:
            temp_data = pair
            # Appends doc ID
            temp_data.append(i)
            data.append(temp_data)
        if i // write_every:
            temp_df = pd.DataFrame(data)
            temp_df.to_csv(file_out_path, sep="\t", index=False, header=None, mode="a")
            del temp_df
            data = []
    temp_df = pd.DataFrame(data)
    temp_df.to_csv(file_out_path, sep="\t", index=False, header=None, mode="a")
    del temp_df

# Reload all data
i2w_in = open("idx_to_word.pickle", "rb")
idx_to_word = pickle.load(i2w_in)

w2i_in = open("word_to_idx.pickle", "rb")
word_to_idx = pickle.load(w2i_in)

freqs = np.load("freqs.npy")
freqs = freqs.tolist()

df = pd.read_csv(file_out_path, sep="\t", header=None)

# Extract data arrays from dataframe
pivot_ids    = df[0].values
target_ids   = df[1].values
doc_ids      = df[2].values

# Shuffle the data
pivot_ids, target_ids, doc_ids = shuffle(pivot_ids, target_ids, doc_ids, random_state=0)

# Hyperparameters
num_docs      = len(np.unique(doc_ids))
vocab_size    = len(word_to_idx.keys())
num_topics    = 25
embed_size    = 256

m = model(num_docs,
                vocab_size,
                num_topics = num_topics,
                embedding_size = embed_size,
                freqs = freqs,
                restore=False)

# Train the model
m.train(pivot_ids,target_ids,doc_ids, len(pivot_ids), 10, context_ids=np.array([dummy, dummy2]), switch_loss_epoch=5)

# Get topic embedding similarities
#idxs = np.array([1,2,3,4,5,6,7,8])
#words, sims = model.get_k_closest(idxs ,in_type="topic", idx_to_word=idx_to_word, k=11)

# Document Mixture
#mix = model.sesh.run(tf.nn.softmax(model.doc_embedding))

# Customer Account Mixture
#mix_custs = model.sesh.run(tf.nn.softmax(model.context_doc_embedding))
from lda2vec import model, utils
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
import pickle
import pyLDAvis

data_path = "skipgrams_clean_full/"
path_to_file = "20_newsgroups.txt"
file_out_path = "20_newsgroups_skipgrams_clean_full.txt"

# Reload all data
i2w_in = open(data_path + "idx_to_word.pickle", "rb")
idx_to_word = pickle.load(i2w_in)

w2i_in = open(data_path + "word_to_idx.pickle", "rb")
word_to_idx = pickle.load(w2i_in)

freqs = np.load(data_path + "freqs.npy")
freqs = freqs.tolist()

embed_matrix = np.load(data_path + "embed_matrix.npy")

df = pd.read_csv(data_path + file_out_path, sep="\t", header=None)

# Extract data arrays from dataframe
pivot_ids = df[0].values
target_ids = df[1].values
doc_ids = df[2].values

# Shuffle the data
pivot_ids, target_ids, doc_ids = shuffle(pivot_ids, target_ids, doc_ids, random_state=0)

# Hyperparameters
num_docs = doc_ids.max() + 1
vocab_size = len(freqs)
num_topics = 20
embed_size = embed_matrix.shape[1]

m = model(num_docs,
          vocab_size,
          num_topics=num_topics,
          embedding_size=embed_size,
          load_embeds=True,
          pretrained_embeddings=embed_matrix,
          freqs=freqs,
          logdir="logdir_180711_0406",
          restore=True
          )

# Train the model
m.train(pivot_ids, target_ids, doc_ids, len(pivot_ids), 500, idx_to_word=idx_to_word, switch_loss_epoch=5)


def generate_ldavis_data():
    """This method will launch a locally hosted session of
    pyLDAvis that will visualize the results of our model
    """
    doc_embed = m.sesh.run(m.doc_embedding)
    topic_embed = m.sesh.run(m.topic_embedding)
    word_embed = m.sesh.run(m.word_embedding)

    # Extract all unique words in order of index 0-vocab_size
    vocabulary = []
    for i in range(vocab_size):
        vocabulary.append(idx_to_word[i])

    # Read in document lengths
    doc_lengths = np.load(data_path + "doc_lengths.npy")

    # utils.py is a direct copy from original authors "topics.py" file
    vis_data = utils.prepare_topics(doc_embed, topic_embed, word_embed, np.array(vocabulary), doc_lengths=doc_lengths,
                                    term_frequency=freqs, normalize=True)

    prepared_vis_data = pyLDAvis.prepare(**vis_data)
    pyLDAvis.show(prepared_vis_data)

# generate_ldavis_data()
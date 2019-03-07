import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import pickle
import pyLDAvis
import os
import random

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def _softmax_2d(x):
    y = x - x.max(axis=1, keepdims=True)
    np.exp(y, out=y)
    y /= y.sum(axis=1, keepdims=True)
    return y


def prob_words(context, vocab, temperature=1.0):
    """ This calculates a softmax over the vocabulary as a function
    of the dot product of context and word.
    """
    dot = np.dot(vocab, context)
    prob = _softmax(dot / temperature)
    return prob


def prepare_topics(weights, factors, word_vectors, vocab, temperature=1.0,
                   doc_lengths=None, term_frequency=None, normalize=False):
    """ Collects a dictionary of word, document and topic distributions.
    Arguments
    ---------
    weights : float array
        This must be an array of unnormalized log-odds of document-to-topic
        weights. Shape should be [n_documents, n_topics]
    factors : float array
        Should be an array of topic vectors. These topic vectors live in the
        same space as word vectors and will be used to find the most similar
        words to each topic. Shape should be [n_topics, n_dim].
    word_vectors : float array
        This must be a matrix of word vectors. Should be of shape
        [n_words, n_dim]
    vocab : list of str
        These must be the strings for words corresponding to
        indices [0, n_words]
    temperature : float
        Used to calculate the log probability of a word. Higher
        temperatures make more rare words more likely.
    doc_lengths : int array
        An array indicating the number of words in the nth document.
        Must be of shape [n_documents]. Required by pyLDAvis.
    term_frequency : int array
        An array indicating the overall number of times each token appears
        in the corpus. Must be of shape [n_words]. Required by pyLDAvis.
    Returns
    -------
    data : dict
        This dictionary is readily consumed by pyLDAVis for topic
        visualization.
    """
    # Map each factor vector to a word
    topic_to_word = []
    msg = "Vocabulary size did not match size of word vectors"
    assert len(vocab) == word_vectors.shape[0], msg
    if normalize:
        word_vectors /= np.linalg.norm(word_vectors, axis=1)[:, None]
    # factors = factors / np.linalg.norm(factors, axis=1)[:, None]
    for factor_vector in factors:
        factor_to_word = prob_words(factor_vector, word_vectors,
                                    temperature=temperature)
        topic_to_word.append(np.ravel(factor_to_word))
    topic_to_word = np.array(topic_to_word)
    msg = "Not all rows in topic_to_word sum to 1"
    assert np.allclose(np.sum(topic_to_word, axis=1), 1), msg
    # Collect document-to-topic distributions, e.g. theta
    doc_to_topic = _softmax_2d(weights)
    msg = "Not all rows in doc_to_topic sum to 1"
    assert np.allclose(np.sum(doc_to_topic, axis=1), 1), msg
    data = {'topic_term_dists': topic_to_word,
            'doc_topic_dists': doc_to_topic,
            'doc_lengths': doc_lengths,
            'vocab': vocab,
            'term_frequency': term_frequency}
    return data


def load_preprocessed_data(data_path, load_embed_matrix=False, shuffle_data=True):
    """Load in all data that was processed via the preprocessing files included
    in this library. Optionally load embedding matrix, if you preprocessed and saved one
    in the data_path. 
    
    Parameters
    ----------
    data_path : TYPE
        Path where all your data is stored. Should be same path you passed the preprocessor when you called save_data()
    load_embed_matrix : bool, optional
        If True, load embedding_matrix.npy file found in data_path.
    shuffle : bool, optional
        If True, it will shuffle the skipgrams dataframe when we load it in. Otherwise, it will leave it in order.
    
    Returns
    -------
    TYPE
        Description
    """
    # Reload all data
    with open(data_path + "/" + "idx_to_word.pickle", "rb") as i2w_in:
        idx_to_word = pickle.load(i2w_in)

    with open(data_path + "/" + "word_to_idx.pickle", "rb") as w2i_in:
        word_to_idx = pickle.load(w2i_in)

    freqs = np.load(data_path + "/" + "freqs.npy")
    freqs = freqs.tolist()

    if load_embed_matrix:
        embed_matrix = np.load(data_path + "/" + "embedding_matrix.npy")

    df = pd.read_csv(data_path + "/skipgrams.txt", sep="\t", header=None)

    # Extract data arrays from dataframe
    pivot_ids = df[0].values
    target_ids = df[1].values
    doc_ids = df[2].values

    if shuffle_data:
        # Shuffle the data
        pivot_ids, target_ids, doc_ids = shuffle(pivot_ids, target_ids, doc_ids, random_state=0)


    if load_embed_matrix:
        return (idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids, embed_matrix)
    else:
        return (idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids)


def generate_ldavis_data(data_path, model, idx_to_word, freqs, vocab_size):
    """This method will launch a locally hosted session of
    pyLDAvis that will visualize the results of our model
    
    Parameters
    ----------
    data_path : str
        Location where your data is stored.
    model : Lda2Vec
        Loaded lda2vec tensorflow model. 
    idx_to_word : dict
        index to word mapping dictionary
    freqs list: 
        Frequencies of each token.
    vocab_size : int
        Total size of your vocabulary
    """

    doc_embed = model.sesh.run(model.mixture.doc_embedding)
    topic_embed = model.sesh.run(model.mixture.topic_embedding)
    word_embed = model.sesh.run(model.w_embed.embedding)

    # Extract all unique words in order of index 0-vocab_size
    vocabulary = []
    for k,v in idx_to_word.items():
        vocabulary.append(v)

    # Read in document lengths
    doc_lengths = np.load(data_path + "/doc_lengths.npy")

    # The prepare_topics function is a direct copy from Chris Moody
    vis_data = prepare_topics(doc_embed, topic_embed, word_embed, np.array(vocabulary), doc_lengths=doc_lengths,
                              term_frequency=freqs, normalize=True)

    prepared_vis_data = pyLDAvis.prepare(**vis_data)
    pyLDAvis.show(prepared_vis_data)

def chunks(n, *args):
    """Yield successive n-sized chunks from l."""
    # From stackoverflow question 312443
    keypoints = []
    for i in range(0, len(args[0]), n):
        keypoints.append((i, i + n))
    random.shuffle(keypoints)
    for a, b in keypoints:
        yield [arg[a: b] for arg in args]
import numpy as np
import pandas as pd
from lda2vec.nlppipe import NlpPipeline
from keras.preprocessing.sequence import skipgrams
from sklearn.utils import shuffle
import pickle
import pyLDAvis
import os


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


def run_preprocessing(texts, data_dir, run_name, min_freq_threshold=10,
                      max_length=100, bad=[], vectors="en_core_web_lg",
                      num_threads=2, token_type="lemma", only_keep_alpha=False,
                      write_every=10000, merge=False):
    """This function abstracts the rest of the preprocessing needed
    to run Lda2Vec in conjunction with the NlpPipeline

    Parameters
    ----------
    texts : TYPE
        Python list of text
    data_dir : TYPE
        directory where your data is held
    run_name : TYPE
        Name of sub-directory to be created that will hold preprocessed data
    min_freq_threshold : int, optional
        If words occur less frequently than this threshold, then purge them from the docs
    max_length : int, optional
        Length to pad/cut off sequences
    bad : list, optional
        List or Set of words to filter out of dataset
    vectors : str, optional
        Name of vectors to load from spacy (Ex. "en", "en_core_web_sm")
    num_threads : int, optional
        Number of threads used in spacy pipeline
    token_type : str, optional
        Type of tokens to keep (Options: "lemma", "lower", "orth")
    only_keep_alpha : bool, optional
        Only keep alpha characters
    write_every : int, optional
        Number of documents' data to store before writing cache to skipgrams file
    merge : bool, optional
        Merge noun phrases or not
    """

    def clean(line):
        return ' '.join(w for w in line.split() if not any(t in w for t in bad))

    # Location for preprocessed data to be stored
    file_out_path = data_dir + "/" + run_name

    if not os.path.exists(file_out_path):

        # Make directory to save data in
        os.makedirs(file_out_path)

        # Remove tokens with these substrings
        bad = set(bad)

        # Preprocess data

        # Convert to unicode (spaCy only works with unicode)
        texts = [str(clean(d)) for d in texts]

        # Process the text, no file because we are passing in data directly
        SP = NlpPipeline(None, max_length, texts=texts,
                         num_threads=num_threads, only_keep_alpha=only_keep_alpha,
                         token_type=token_type, vectors=vectors, merge=merge)

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
                temp_df.to_csv(file_out_path + "/skipgrams.txt", sep="\t", index=False, header=None, mode="a")
                del temp_df
                data = []
            if i % 500 == 0:
                print("step", i, "of", num_examples)
        temp_df = pd.DataFrame(data)
        temp_df.to_csv(file_out_path + "/skipgrams.txt", sep="\t", index=False, header=None, mode="a")
        del temp_df

        # Save embed matrix
        np.save(file_out_path + "/embed_matrix", embed_matrix)
        # Save the doc lengths to be used later, also, purge those that didnt make it into skipgram function
        np.save(file_out_path + "/doc_lengths", np.delete(doc_lengths, np.array(purged_docs)))
        # Save frequencies to file
        np.save(file_out_path + "/freqs", freqs)
        # Save vocabulary dictionaries to file
        idx_to_word_out = open(file_out_path + "/" + "idx_to_word.pickle", "wb")
        pickle.dump(SP.idx_to_word, idx_to_word_out)
        idx_to_word_out.close()
        word_to_idx_out = open(file_out_path + "/" + "word_to_idx.pickle", "wb")
        pickle.dump(SP.word_to_idx, word_to_idx_out)
        word_to_idx_out.close()


def load_preprocessed_data(data_path, run_name):
    # Set file out path
    file_out_path = data_path + "/" + run_name

    # Reload all data
    i2w_in = open(file_out_path + "/" + "idx_to_word.pickle", "rb")
    idx_to_word = pickle.load(i2w_in)

    w2i_in = open(file_out_path + "/" + "word_to_idx.pickle", "rb")
    word_to_idx = pickle.load(w2i_in)

    freqs = np.load(file_out_path + "/" + "freqs.npy")
    freqs = freqs.tolist()

    embed_matrix = np.load(file_out_path + "/" + "embed_matrix.npy")

    df = pd.read_csv(file_out_path + "/skipgrams.txt", sep="\t", header=None)

    # Extract data arrays from dataframe
    pivot_ids = df[0].values
    target_ids = df[1].values
    doc_ids = df[2].values

    # Shuffle the data
    pivot_ids, target_ids, doc_ids = shuffle(pivot_ids, target_ids, doc_ids, random_state=0)

    # Hyperparameters
    num_docs = doc_ids.max() + 1
    vocab_size = len(freqs)
    embed_size = embed_matrix.shape[1]

    return (idx_to_word, word_to_idx, freqs, embed_matrix, pivot_ids,
            target_ids, doc_ids, num_docs, vocab_size, embed_size)


def generate_ldavis_data(data_path, run_name, model,
                         idx_to_word, freqs, vocab_size):
    """This method will launch a locally hosted session of
    pyLDAvis that will visualize the results of our model
    """
    doc_embed = model.sesh.run(model.doc_embedding)
    topic_embed = model.sesh.run(model.topic_embedding)
    word_embed = model.sesh.run(model.word_embedding)

    # Extract all unique words in order of index 0-vocab_size
    vocabulary = []
    for i in range(vocab_size):
        vocabulary.append(idx_to_word[i])

    # Read in document lengths
    doc_lengths = np.load(data_path + "/" + run_name + "/" + "doc_lengths.npy")

    # The prepare_topics function is a direct copy from Chris Moody
    vis_data = prepare_topics(doc_embed, topic_embed, word_embed, np.array(vocabulary), doc_lengths=doc_lengths,
                              term_frequency=freqs, normalize=True)

    prepared_vis_data = pyLDAvis.prepare(**vis_data)
    pyLDAvis.show(prepared_vis_data)
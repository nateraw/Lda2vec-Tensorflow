import numpy as np
import tensorflow as tf


def _orthogonal_matrix(shape):
    # Stolen from blocks:
    # github.com/mila-udem/blocks/blob/master/blocks/initialization.py
    M1 = np.random.randn(shape[0], shape[0])
    M2 = np.random.randn(shape[1], shape[1])

    # QR decomposition of matrix with entries in N(0, 1) is random
    Q1, R1 = np.linalg.qr(M1)
    Q2, R2 = np.linalg.qr(M2)
    # Correct that NumPy doesn"t force diagonal of R to be non-negative
    Q1 = Q1 * np.sign(np.diag(R1))
    Q2 = Q2 * np.sign(np.diag(R2))
    n_min = min(shape[0], shape[1])
    return np.dot(Q1[:, :n_min], Q2[:n_min, :])


class EmbedMixture():
    def __init__(self, n_documents, n_topics, n_dim, temperature=1.0,
                 W_in=None, factors_in=None, name=""):
        self.n_documents = n_documents
        self.temperature = temperature
        self.name = name
        # Sets the dropout value
        #self.dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        scalar = 1 / np.sqrt(n_documents + n_topics)


        # self.W in original
        if not isinstance(W_in, np.ndarray):
            self.Doc_Embedding = tf.Variable(tf.random_normal([n_documents, n_topics], mean=0, stddev=50 * scalar),
                             name=self.name+ "_" +"doc_embedding")
        else:
            # Initialize the weights as a constant
            init = tf.constant(W_in)
            # Convert the weights to a tensorflow variable
            self.Doc_Embedding = tf.get_variable(self.name+ "_" +"doc_embedding", initializer=init)


        with tf.name_scope(self.name+ "_" +"Topics"):

            # self.factors in original... Unnormalized embedding weights
            if not isinstance(factors_in, np.ndarray):
                self.topic_embedding = tf.get_variable(self.name+ "_" +"topic_embedding", shape=[n_topics, n_dim], dtype=tf.float32,
                                                   initializer=tf.orthogonal_initializer(gain=scalar))
            else:
                # Initialize the weights as a constant
                init = tf.constant(factors_in)
                # Convert the weights to a tensorflow variable
                self.topic_embedding = tf.get_variable(self.name+ "_" +"topic_embedding", initializer=init)       
            #self.topic_embedding = tf.nn.dropout(topic_embedding, self.dropout, name="topic_dropout")

    def __call__(self, doc_ids=None, update_only_docs=False):
        # Get proportions from function below this one
        proportions = self.proportions(doc_ids, softmax=True)

        # multiply proportions by the factors_in
        w_sum = tf.matmul(proportions, self.topic_embedding, name=self.name+ "_" +"docs_mul_topics")

        return w_sum

    def proportions(self, doc_ids=None, softmax=False):
        # Given an array of document indices, return a vector for
        # each document of just the unnormalized topic weights
        if doc_ids == None:
            w = self.Doc_Embedding
        else:
            w = tf.nn.embedding_lookup(self.Doc_Embedding, doc_ids, name=self.name+ "_" +"doc_proportions")

        if softmax:
            return tf.nn.softmax(w / self.temperature)
        else:
            return w

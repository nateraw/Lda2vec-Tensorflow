import numpy as np
import tensorflow as tf



class EmbedMixture():
    def __init__(self, n_documents, n_topics, n_dim, temperature=1.0,
                 W_in=None, factors_in=None):
        self.n_documents = n_documents
        self.temperature = temperature
        # Sets the dropout value
        self.dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        scalar = 1 / np.sqrt(n_documents + n_topics)

        # self.W in original
        self.Doc_Embedding = tf.Variable(
            tf.random_normal([n_documents, n_topics], mean=0, stddev=50 * scalar),
            name="Doc_Embedding")

        with tf.name_scope("Topics"):

            # self.factors in original... Unnormalized embedding weights
            self.topic_embedding = tf.get_variable("topic_embedding", shape=[n_topics, n_dim], dtype=tf.float32,
                                                   initializer=tf.orthogonal_initializer(gain=scalar))
            #self.topic_embedding = tf.nn.dropout(topic_embedding, self.dropout, name="topic_dropout")

    def __call__(self, doc_ids=None, update_only_docs=False):
        # Get proportions from function below this one
        proportions = self.proportions(doc_ids, softmax=True)

        # multiply proportions by the factors_in
        w_sum = tf.matmul(proportions, self.topic_embedding, name="docs_mul_topics")

        return w_sum

    def proportions(self, doc_ids=None, softmax=False):
        # Given an array of document indices, return a vector for
        # each document of just the unnormalized topic weights
        if doc_ids == None:
            w = self.Doc_Embedding
        else:
            w = tf.nn.embedding_lookup(self.Doc_Embedding, doc_ids, name="doc_proportions")

        if softmax:
            return tf.nn.softmax(w / self.temperature)
        else:
            return w

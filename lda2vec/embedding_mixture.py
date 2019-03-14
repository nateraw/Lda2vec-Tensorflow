import numpy as np
import tensorflow as tf


def _orthogonal_matrix(shape):
    M1 = np.random.randn(shape[0], shape[0])
    M2 = np.random.randn(shape[1], shape[1])
    Q1, R1 = np.linalg.qr(M1)
    Q2, R2 = np.linalg.qr(M2)
    Q1 = Q1 * np.sign(np.diag(R1))
    Q2 = Q2 * np.sign(np.diag(R2))
    n_min = min(shape[0], shape[1])
    return np.dot(Q1[:, :n_min], Q2[:n_min, :])


class EmbedMixture:

    def __init__(self, n_documents, n_topics, n_dim, temperature=1.0, W_in=None, factors_in=None, name=''):
        self.n_documents = n_documents
        self.temperature = temperature
        self.name = name
        scalar = 1 / np.sqrt(n_documents + n_topics)
        
        self.doc_embedding = tf.Variable(tf.random_normal([n_documents, n_topics], mean=0, stddev=50 * scalar),
                                         name='doc_embedding') if W_in is None else W_in

        self.topic_embedding = tf.get_variable('topic_embedding', shape=[n_topics, n_dim],
                                               dtype=tf.float32,
                                               initializer=tf.orthogonal_initializer(gain=scalar)) if factors_in is None else factors_in


    def __call__(self, doc_ids=None, update_only_docs=False, softmax=True):
        proportions = self.proportions(doc_ids, softmax=softmax)
        w_sum = tf.matmul(proportions, self.topic_embedding, name='docs_mul_topics')
        return w_sum

    def proportions(self, doc_ids=None, softmax=False):
        if doc_ids == None:
            w = self.doc_embedding
        else:
            w = tf.nn.embedding_lookup(self.doc_embedding, doc_ids, name='doc_proportions')
        if softmax:
            return tf.nn.softmax(w / self.temperature)
        else:
            return w
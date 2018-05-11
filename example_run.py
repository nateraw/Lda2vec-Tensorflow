from lda2vec import Lda2vec
import tensorflow as tf
from preprocessing_class import Lda2vec_Preprocessor

# This file has 278677 documents
path_to_file  = "example_text_sm.txt"
file_out_path = "lda2vec_data_large.txt"

glove_embed_file = "glove.6B.100d.txt"

# Instantiate preprocessor for lda2vec preprocessing
P = Lda2vec_Preprocessor(path_to_file,
                         file_out_path,
                         load_pretrained_embeddings=True,
                         compute_skipgrams=True,
                         glove_embeddings_file=glove_embed_file,
                         extract_data=True)

# Get training data needed to feed into network
pivot_words, target_words, doc_ids = P.get_training_data()

# Set the number of topics to cluster documents into
num_topics = 15


# Instantiate LDA2Vec
model = Lda2vec(P.num_unique_documents,
                P.vocab_size,
                num_topics = 10,
                freqs = P.freqs,
                load_embeds=True,
                pretrained_embeddings=P.embed_matrix,
                embedding_size = P.embedding_dim,
                restore=False)

# Train the model
model.train(pivot_words,target_words,doc_ids, len(pivot_words), 2)

# Analyze Document Proportions - shape [num_documents, num_topics]
mix = model.sesh.run(tf.nn.softmax(model.doc_embedding))

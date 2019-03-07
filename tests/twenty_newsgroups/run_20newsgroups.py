from lda2vec import utils, model
import os



data_path  = "data/clean_data"
num_topics = 20
num_epochs = 200

# Load data from file - Do not load embeddings
#(idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids) = utils.load_preprocessed_data(data_path)

# Load data from file - Load embeddings
(idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids, embed_matrix) = utils.load_preprocessed_data(data_path, load_embed_matrix=True)

# Hyperparameters
num_docs = doc_ids.max() + 1
vocab_size = len(freqs)
embed_size = embed_matrix.shape[1]


m = model(num_docs,
          vocab_size,
          num_topics,
          embedding_size=embed_size,
          w_in=embed_matrix,
          freqs=freqs,
          batch_size = 4096,
          save_graph_def=False)

m.train(pivot_ids,target_ids,doc_ids, len(pivot_ids), num_epochs, idx_to_word=idx_to_word,  switch_loss_epoch=5)

utils.generate_ldavis_data(data_path, m, idx_to_word, freqs, vocab_size)
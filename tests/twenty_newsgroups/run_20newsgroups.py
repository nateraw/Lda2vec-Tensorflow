from lda2vec import utils, model

data_path  = "data/clean_data"
num_topics = 20
num_epochs = 20

# Load data from file - Do not load embeddings
(idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids) = utils.load_preprocessed_data(data_path)

# Load data from file - Load embeddings
# (idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids, embed_matrix) = utils.load_preprocessed_data(data_path, load_embed_matrix=True)


# Hyperparameters
num_docs = doc_ids.max() + 1
vocab_size = len(freqs)
embed_size = 128 # If you loaded the embed matrix, use embed_matrix.shape[1]


m = model(num_docs,
          vocab_size,
          num_topics=num_topics,
          embedding_size=embed_size,
          load_embeds=False, # True if embed_matrix loaded
          pretrained_embeddings=None, # embed_matrix if you loaded it
          freqs=freqs)

m.train(pivot_ids,target_ids,doc_ids, len(pivot_ids), num_epochs, idx_to_word=idx_to_word,  switch_loss_epoch=5)

utils.generate_ldavis_data(data_path, run_name, m, idx_to_word, freqs, vocab_size)
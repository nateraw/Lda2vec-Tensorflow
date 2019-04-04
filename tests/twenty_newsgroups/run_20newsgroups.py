from lda2vec import utils, model
import numpy as np

# Path to preprocessed data
data_path  = "data/clean_data"
# Whether or not to load saved embeddings file
load_embeds = True

# Load data from files
(idx_to_word, word_to_idx, freqs, pivot_ids,
 target_ids, doc_ids, embed_matrix) = utils.load_preprocessed_data(data_path, load_embed_matrix=load_embeds)

# Number of unique documents
num_docs = len(np.unique(doc_ids))
# Number of unique words in vocabulary (int)
vocab_size = embed_matrix.shape[0] 
# Embed layer dimension size
# If not loading embeds, change 128 to whatever size you want.
embed_size = embed_matrix.shape[1] if load_embeds else 128
# Number of topics to cluster into
num_topics = 20
# Epoch that we want to "switch on" LDA loss
switch_loss_epoch = 5
# Pretrained embeddings
pretrained_embeddings = embed_matrix if load_embeds else None
# If True, save logdir, otherwise don't
save_graph = True
num_epochs = 200
batch_size = 512 #4096

# Initialize the model
m = model(num_docs,
          vocab_size,
          num_topics,
          embedding_size=embed_size,
          pretrained_embeddings=pretrained_embeddings,
          freqs=freqs,
          batch_size = batch_size,
          save_graph_def=save_graph)

# Train the model
m.train(pivot_ids,
        target_ids,
        doc_ids,
        len(pivot_ids),
        num_epochs,
        idx_to_word=idx_to_word,
        switch_loss_epoch=switch_loss_epoch)

# Visualize topics with pyldavis
utils.generate_ldavis_data(data_path, m, idx_to_word, freqs, vocab_size)
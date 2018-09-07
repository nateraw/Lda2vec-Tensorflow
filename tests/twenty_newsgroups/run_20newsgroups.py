from lda2vec import utils, model

data_path  = "data"
run_name   = "my_run"
num_topics = 20
num_epochs = 20

(idx_to_word, word_to_idx, freqs, embed_matrix, pivot_ids,
 target_ids, doc_ids, num_docs, vocab_size, embed_size) = utils.load_preprocessed_data(data_path, run_name)

m = model(num_docs,
          vocab_size,
          num_topics=num_topics,
          embedding_size=embed_size,
          load_embeds=True,
          pretrained_embeddings=embed_matrix,
          freqs=freqs)

m.train(pivot_ids,target_ids,doc_ids, len(pivot_ids), num_epochs, idx_to_word=idx_to_word,  switch_loss_epoch=5)

utils.generate_ldavis_data(data_path, run_name, m, idx_to_word, freqs, vocab_size)
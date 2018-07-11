# Lda2vec-Tensorflow
Tensorflow 1.5 implementation of Chris Moody's Lda2vec, adapted from @meereeum

## Usage

### Preprocessing

The preprocessing is all done through the "nlppipe.py" file. Using SpaCy,
we have added a lot of functionality. We can pad/cut off our sentences,
merge noun phrases, use parallel processing, and load pretrained vectors.

At the most basic level, if you would like to get your data processed for lda2vec,
you can do the following:

```python
# First, init the processor and pass some parameters
SP = NlpPipeline(path_to_file, # Line delimited input file with no header
                 50, # Number of tokens to limit/pad each sequence to
                 merge=True, # When true, we merge noun phrases
                 num_threads=8) # Uses 8 cores
# Then, you can compute the embed matrix
SP._compute_embed_matrix(random=True) # Random init embed matrix

# Convert data to word2vec indexes (from SpaCy hashes)
SP.convert_data_to_word2vec_indexes()

# I choose to trim the 0s off of my sequences
SP.trim_zeros_from_idx_data()
```

##

##### This will leave you with the following variables:
SP.embed_matrix - Embedding matrix of shape [vocab_size, embedding_size]

SP.idx_data -  Your tokenized data of shape [num_documents, ?]

SP.idx_to_word - index to word mapping of vocabulary

SP.word_to_idx - word to index mapping of vocabulary

SP.freqs - Frequencies of each word in order of the embed matrix.


### Initializing the model
Once you have preprocessed the data, you can pass data from the
preprocessing class directly into the model init. If you use your own
data, you can just pass integers to the num_unique_documents, vocab_size,
and embedding_size parameters.

Also, you don't have to initialize the embedding matrix yourself,
it will init them randomly if you erase the pretrained_embeddings parameter shown here:
```python
model = Lda2vec(num_docs,
                vocab_size,
                num_topics = 10,
                freqs = freqs,
                load_embeds=True,
                pretrained_embeddings=SP.embed_matrix,
                embedding_size = embedding_size,
                restore=False)
```

### Training the model
To train the model, it is as easy as passing our python lists of pivot words, target
words and document ids to the model.train function. Additionally, the model.train
function is looking for the number of data points, so we do this by checking
the length of the input pivot words (pivot words, target words, and doc ids should all
be python lists of the same size). The last parameter is the number of epochs to train for.
```python
model.train(pivot_words, target_words, doc_ids, len(pivot_words), 2)
```

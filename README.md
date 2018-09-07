# Lda2vec-Tensorflow
Tensorflow 1.5 implementation of Chris Moody's Lda2vec, adapted from @meereeum

## Usage
### Installation
Currently, the setup.py and the pip install are both not working!
Unfortunately, I suggest you unpack the files yourself, for now.
I am actively looking for help fixing that problem!
### Preprocessing

The preprocessing is all done through the "nlppipe.py" file. Using SpaCy,
we have added a lot of functionality. We can pad/cut off our sentences,
merge noun phrases, use parallel processing, and load pretrained vectors.

At the most basic level, if you would like to get your data processed for lda2vec,
you can do the following:

```python
data_dir = "data"
run_name = "my_run"

# Python list of your text
texts = ["list of your text here", ..., "your text here"]

# Run preprocessing, limiting/padding documents to 100 tokens
utils.run_preprocessing(texts, data_dir, run_name, max_length=100, vectors="en_core_web_sm")
```

When you run the twenty newsgroups example, it will create a directory tree that looks like this:
```bash
├── my_project
│   ├── data
│   │   ├── 20_newsgroups.txt
│   │   └── my_run
│   │       ├── doc_lengths.npy
│   │       ├── embed_matrix.npy
│   │       ├── freqs.npy
│   │       ├── idx_to_word.pickle
│   │       ├── skipgrams.txt
│   │       └── word_to_idx.pickle
│   ├── load_20newsgroups.py
│   └── run_20newsgroups.py
```

### Using the Model

To run the model, pass the same data_path and run_name to the
load_preprocessed_data function and then use that data to instantiate and train the model.

```python
data_dir = "data"
run_name = "my_run"
num_topics = 20
num_epochs = 20

# Load preprocessed data
idx_to_word, word_to_idx, freqs, embed_matrix, pivot_ids,
target_ids, doc_ids, num_docs, vocab_size, embed_size) = utils.load_preprocessed_data(data_dir, run_name)

# Instantiate the model
m = model(num_docs,
          vocab_size,
          num_topics = num_topics,
          embedding_size = embed_size,
          load_embeds=True,
          pretrained_embeddings=embed_matrix,
          freqs = freqs)

# Train the model
m.train(pivot_ids,target_ids,doc_ids, len(pivot_ids), num_epochs, idx_to_word = idx_to_word,  switch_loss_epoch=5)
```

### Visualizing the Results
We can now visualize the results of our model using pyLDAvis:

```python
utils.generate_ldavis_data(data_path, run_name, m, idx_to_word, freqs, vocab_size)
```
This will launch pyLDAvis in your browser, allowing you to visualize your results like this:

![alt text](https://github.com/nateraw/Lda2vec-Tensorflow/blob/master/pyLDAvis_results.png)
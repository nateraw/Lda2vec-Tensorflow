# Lda2vec-Tensorflow
Tensorflow 1.5 implementation of Chris Moody's Lda2vec, adapted from @meereeum

## Usage
### Installation
`pip install lda2vec`

### Preprocessing

The preprocessing is all done through the "nlppipe.py" file. Using SpaCy,
we have added a lot of functionality. We can pad/cut off our sentences,
merge noun phrases, use parallel processing, and load pretrained vectors.

At the most basic level, if you would like to get your data processed for lda2vec,
you can do the following:

```python
# Data directory
data_dir ="data"
# Where to save preprocessed data
clean_data_dir = "data/clean_data"
# Name of input file. Should be inside of data_dir
input_file = "20_newsgroups.txt"
# Should we load glove vectors from file?
load_glove_vecs = False


# Read in data file
df = pd.read_csv(data_dir+"/"+input_file, sep="\t")

# Initialize a preprocessor
P = Preprocessor(df, "texts", max_features=30000)

# Run the preprocessing on your dataframe
P.preprocess()

# Optionally, you can load and save glove vectors from file
if load_glove_vecs:
    embedding_matrix = P.load_glove("PATH/TO/GLOVE/glove.6B.300d.txt")
    P.save_data(clean_data_dir, embedding_matrix=embedding_matrix)
else:
    P.save_data(clean_data_dir)
```

When you run the twenty newsgroups example, it will create a directory tree that looks like this:
```bash
├── my_project
│   ├── data
│   │   ├── 20_newsgroups.txt
│   │   └── clean_data_dir
│   │       ├── doc_lengths.npy
│   │       ├── embedding_matrix.npy
│   │       ├── freqs.npy
│   │       ├── idx_to_word.pickle
│   │       ├── skipgrams.txt
│   │       └── word_to_idx.pickle
│   ├── load_20newsgroups.py
│   └── run_20newsgroups.py
```

### Using the Model

To run the model, pass the same data_path to the
load_preprocessed_data function and then use that data to instantiate and train the model.

```python
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

m.train(pivot_ids,target_ids,doc_ids, len(pivot_ids), num_epochs, idx_to_word=idx_to_word,  switch_loss_epoch=5)```
```

### Visualizing the Results
We can now visualize the results of our model using pyLDAvis:
```python
utils.generate_ldavis_data(data_path, m, idx_to_word, freqs, vocab_size)
```
This will launch pyLDAvis in your browser, allowing you to visualize your results like this:

![alt text](https://github.com/nateraw/Lda2vec-Tensorflow/blob/master/pyLDAvis_results.png)

# Lda2vec-Tensorflow
Tensorflow 1.5 implementation of Chris Moody's Lda2vec, adapted from @meereeum

## Note
This algorithm is very much so a research algorithm. It doesn't always work so well, and you have to train it for a long time. As the author noted in the paper, most of the time normal LDA will work better.

Note that you should run this algorithm for **at least 20 epochs** before expecting to see any results. The algorithm is meant to run for a very long time. 

## Usage
### Installation
Clone the repo and run `python setup.py install` to install the package as is or run `python setup.py develop` to make your own edits. 

You can also just `pip install lda2vec` (Last updated 2/28/19)

### Preprocessing

The preprocessing is all done through the "nlppipe.py" file using Spacy. Feel free to use your own preprocessing, if you like.

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
# Spacy model to use
nlp = "en_core_web_lg" # "en", "en_core_web_sm", or any other Spacy Language model you have
# Number of tokens to cut documents off at
maxlen = 100
# Maximum number of words to keep in your vocabulary
max_features = 30000

# Read in data file
df = pd.read_csv(data_dir+"/"+input_file, sep="\t")

# Initialize a preprocessor
P = Preprocessor(df, # Data loaded into dataframe. Each row has a document.
                "texts", # Name of text column in your dataframe
                max_features=max_features,
                maxlen=maxlen,
                nlp=nlp)

# Run the preprocessing on your dataframe
P.preprocess()

# Optionally, you can load and save glove vectors from file
if load_glove_vecs:
    embedding_matrix = P.load_glove("PATH/TO/GLOVE/glove.6B.300d.txt")
    P.save_data(clean_data_dir, embedding_matrix=embedding_matrix)
else:
    P.save_data(clean_data_dir)
```

When you run the twenty newsgroups preprocessing example, it will create a directory tree that looks like this:
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

# Initialize the model
m = model(num_docs, # Number of documents in your corpus
          vocab_size, # Number of unique words in your vocabulary
          num_topics=num_topics, # Number of topics to learn
          embedding_size=embed_size, # Embedding dimension size
          load_embeds=False, # True if embed_matrix loaded
          pretrained_embeddings=None, # numpy.ndarray embed_matrix if you loaded it
          freqs=freqs) # Python list of shape (vocab_size,). Frequencies of each token, same order as embed matrix mappings.

# Train the model
m.train(pivot_ids,target_ids,
        doc_ids,
        len(pivot_ids),
        num_epochs,
        switch_loss_epoch = 5
        idx_to_word=idx_to_word)
```

### Visualizing the Results
We can now visualize the results of our model using pyLDAvis:
```python
utils.generate_ldavis_data(data_path, m, idx_to_word, freqs, vocab_size)
```
This will launch pyLDAvis in your browser, allowing you to visualize your results like this:

![alt text](https://github.com/nateraw/Lda2vec-Tensorflow/blob/master/pyLDAvis_results.png)

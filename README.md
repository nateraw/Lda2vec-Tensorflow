# Lda2vec-Tensorflow
Tensorflow 1.5 implementation of Chris Moody's Lda2vec, adapted from @meereeum

## Note
This algorithm is very much so a research algorithm. It doesn't always work so well, and you have to train it for a long time. As the author noted in the paper, most of the time normal LDA will work better.

Note that you should run this algorithm for **at least 100 epochs** before expecting to see any results. The algorithm is meant to run for a very long time. 

## Usage
### Installation
**Warning**- You may have to install dependencies manually before being able to use the package. Requirements can be found [here](https://github.com/nateraw/Lda2vec-Tensorflow/issues/27). If you install these on a clean environment, you should be good to go. I am seeking help on this issue.

Clone the repo and run `python setup.py install` to install the package as is or run `python setup.py develop` to make your own edits. 

You can also just `pip install lda2vec` (Last updated 3/13/19)

### Pretrained Embeddings
This repo can load a wide variety of pretrained embedding files (see [nlppipe.py](https://github.com/nateraw/Lda2vec-Tensorflow/blob/5d399a3f21dd3e9a2e84a6220d5f9e3147a3591b/lda2vec/nlppipe.py#L115) for more info). The examples are all using GloVe embeddings. You can download them from [here](https://github.com/stanfordnlp/GloVe). 

### Preprocessing

The preprocessing is all done through the "nlppipe.py" file using Spacy. Feel free to use your own preprocessing, if you like.

At the most basic level, if you would like to get your data processed for lda2vec,
you can do the following:

```python
import pandas as pd
from lda2vec.nlppipe import Preprocessor

# Data directory
data_dir ="data"
# Where to save preprocessed data
clean_data_dir = "data/clean_data"
# Name of input file. Should be inside of data_dir
input_file = "20_newsgroups.txt"
# Should we load pretrained embeddings from file
load_embeds = True

# Read in data file
df = pd.read_csv(data_dir+"/"+input_file, sep="\t")

# Initialize a preprocessor
P = Preprocessor(df, "texts", max_features=30000, maxlen=10000, min_count=30)

# Run the preprocessing on your dataframe
P.preprocess()

# Load embeddings from file if we choose to do so
if load_embeds:
    # Load embedding matrix from file path - change path to where you saved them
    embedding_matrix = P.load_glove("PATH/TO/GLOVE/glove.6B.300d.txt")
else:
    embedding_matrix = None

# Save data to data_dir
P.save_data(clean_data_dir, embedding_matrix=embedding_matrix)
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

# Path to preprocessed data
data_path  = "data/clean_data"
# Whether or not to load saved embeddings file
load_embeds = True

# Load data from files
(idx_to_word, word_to_idx, freqs, pivot_ids,
 target_ids, doc_ids, embed_matrix) = utils.load_preprocessed_data(data_path, load_embed_matrix=load_embeds)

# Number of unique documents
num_docs = doc_ids.max() + 1
# Number of unique words in vocabulary (int)
vocab_size = len(freqs)
# Embed layer dimension size
# If not loading embeds, change 128 to whatever size you want.
embed_size = embed_matrix.shape[1] if load_embeds else 128
# Number of topics to cluster into
num_topics = 20
# Amount of iterations over entire dataset
num_epochs = 200
# Batch size - Increase/decrease depending on memory usage
batch_size = 4096
# Epoch that we want to "switch on" LDA loss
switch_loss_epoch = 0
# Pretrained embeddings value
pretrained_embeddings = embed_matrix if load_embeds else None
# If True, save logdir, otherwise don't
save_graph = True


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
```

### Visualizing the Results
We can now visualize the results of our model using pyLDAvis:
```python
utils.generate_ldavis_data(data_path, m, idx_to_word, freqs, vocab_size)
```
This will launch pyLDAvis in your browser, allowing you to visualize your results like this:

![alt text](https://github.com/nateraw/Lda2vec-Tensorflow/blob/master/pyLDAvis_results.png)

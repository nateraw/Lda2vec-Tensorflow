# Lda2vec-Tensorflow
Tensorflow 1.5 implementation of Chris Moody's Lda2vec, adapted from @meereeum


## Usage

### Preprocessing

First, you will pass a line delmited text file to the lda2vec processor
class. To reiterate, you will have each document in a text file separated by
a new line. I have included an example file called "example_text_sm.txt".

Once preprocessed, it will save tab delimited input data to file_out_path
so that you can run lda2vec on the same data faster.

So, when you first start, you will run the processor like this:
```python
P = Lda2vec_Preprocessor(path_to_file,
                         file_out_path,
                         load_pretrained_embeddings=True,
                         compute_skipgrams=True,
                         glove_embeddings_file=glove_embed_file,
                         extract_data=True)
```

Here, your path to file should be the path to the line delmited documents.
The file out path should be a txt file that will hold your lda2vec input data.
If you want to use pretrained embeddings from glove (recommended), you pass
load_pretrained_embeddings= True and pass the location of the glove embeddings to
glove_embeddings_file parameter.

Compute_skipgrams should be True if you haven't
run the processor yet! Once you run it and file_out_path exists, change this parameter
to False:

```python
P = Lda2vec_Preprocessor(path_to_file,
                         file_out_path,
                         load_pretrained_embeddings=True,
                         compute_skipgrams=False, # if file out path exists!
                         glove_embeddings_file=glove_embed_file,
                         extract_data=True)
```

### Initializing the model
Once you have preprocessed the data, you can pass data from the
preprocessing class directly into the model init. If you use your own
data, you can just pass integers to the num_unique_documents, vocab_size,
and embedding_size parameters.

Also, you don't have to initialize the embedding matrix yourself,
it will init them randomly if you erase the pretrained_embeddings parameter shown here:
```python
model = Lda2vec(P.num_unique_documents,
                P.vocab_size,
                num_topics = 10,
                freqs = P.freqs,
                load_embeds=True,
                pretrained_embeddings=P.embed_matrix,
                embedding_size = P.embedding_dim,
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

## Notes

This implementation still needs some work, and additional functionality is coming.

Also, I know the loss is negative. However, it works...trust me!
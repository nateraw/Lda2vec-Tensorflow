from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import spacy
from keras.preprocessing.sequence import skipgrams
import pickle
from tqdm import tqdm
import os

class Preprocessor:
    def __init__(self, df, textcol, max_features=30000, maxlen=None, window_size=5, nlp="en_core_web_sm",
                 bad=set(["ax>", '`@("', '---', '===', '^^^', "AX>", "GIZ"]), token_type="lower", min_count=None):
        """Summary
        
        Args:
            df (TYPE): Dataframe that has your text in a column
            textcol (TYPE): Name of the column in your dataframe that holds the text
            max_features (int, optional): Maximum number of unique words to keep
            maxlen (int, optional): Maximum length of a document. Will cut off documents
                at this length after tokenization but before computing skipgram pairs.
            window_size (int, optional): size of sampling windows (technically half-window). 
                The window of a word w_i will be  [i - window_size, i + window_size+1].
            nlp (str, optional): Spacy model to load (ex. "en", "en_core_web_sm", "en_core_web_lg", 
                or some custom one you have)
            bad (TYPE, optional): Set of known bad characters for your dataset that you'd like to remove.
            token_type (str, optional): Type of token to keep. Options are "lower", "lemma". If you pass something
                that isn't in these options, you will get the original text back (i.e. not lowercased or anything)
            min_count (int, optional): Will remove words below this frequency if an integer is given. Defaults to not
                deleting anything
        """
        self.df = df
        self.textcol = textcol
        self.bad = bad
        self.token_type = token_type
        self.maxlen = maxlen
        self.max_features = max_features
        self.window_size = window_size
        self.min_count = min_count

        # Here we disable parts of spacy's pipeline - REALLY improves speed.
        self.nlp = spacy.load(nlp, disable = ['ner', 'parser'])
        self.nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS,
                                                       spacy.attrs.IS_STOP)

    def clean(self, line):
        return ' '.join(w for w in line.split() if not any(t in w for t in self.bad))

    def tokenize_and_process(self):

        # Get texts into python list
        texts = self.df[self.textcol].values.tolist()

        # Do some quick cleaning of the texts
        texts = [str(self.clean(d)) for d in texts]

        # Get number of docs supplied
        self.num_docs = len(self.df)

        # Init list for holding clean text
        self.texts_clean = []

        print("\n---------- Tokenizing Texts ----------")
        # Iterate over all uncleaned documents
        for i, doc in tqdm(enumerate(self.nlp.pipe(texts, n_threads=4))):
            # Variable for holding cleaned tokens (to be joined later)
            doc_texts = []
            for token in doc:
                # Some options for you - TODO pass attrs dictionary
                #if not token.like_email and not token.like_url and not token.is_punct and not token.like_num and token.is_alpha:
                if token.is_alpha and not token.is_stop:
                    if self.token_type == "lemma":
                        if token.lemma_ == "-PRON-":
                            doc_texts.append(token.lower_)
                        else:
                            doc_texts.append(token.lemma_)
                    elif self.token_type=="lower":
                        doc_texts.append(token.lower_)
                    else:
                        doc_texts.append(token.text)
            self.texts_clean.append(" ".join(doc_texts))

        # Init a tokenizer and fit it with our cleaned texts
        self.tokenizer = Tokenizer(self.max_features, filters="", lower=False)
        self.tokenizer.fit_on_texts(self.texts_clean)
        self.tokenizer.word_index["<UNK>"] = 0
        self.tokenizer.word_docs["<UNK>"] = 0
        self.tokenizer.word_counts["<UNK>"] = 0

        # This chunk handles removing words with counts less than min_count
        if self.min_count != None:
            # Get all the words to remove
            words_to_rm = [w for w,c in self.tokenizer.word_counts.items() if c < self.min_count and w != "<UNK>"]
            print("Removing {0} low frequency tokens out of {1} total tokens".format(len(words_to_rm), len(self.tokenizer.word_counts)))
            # Iterate over those words and remove them from the necessary Tokenizer attributes
            for w in words_to_rm:
                del self.tokenizer.word_index[w]
                del self.tokenizer.word_docs[w]
                del self.tokenizer.word_counts[w]

        # Get the idx data from the tokenizer
        self.idx_data = self.tokenizer.texts_to_sequences(self.texts_clean)

        # Limit the data to be maxlen
        if self.maxlen != None:
            for i, d in enumerate(self.idx_data):
                self.idx_data[i] = d[:self.maxlen]

    def get_supplemental_data(self):

        # Get idx to word from keras tokenizer
        self.word_to_idx = self.tokenizer.word_index

        # Flip idx to word to get word_to_idx
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        # Vocab size should be at most max_features, but will default to len(word_index) if less than that
        self.vocab_size = min(self.max_features, len(self.idx_to_word))

        # Init empty list to hold frequencies
        self.freqs = []
        for i in range(self.vocab_size):
            token = self.idx_to_word[i]
            self.freqs.append(self.tokenizer.word_counts[token])
    
    def load_glove(self, EMBEDDING_FILE):
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        # word_index = tokenizer.word_index
        nb_words = self.vocab_size
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in self.word_to_idx.items():
            if i >= nb_words: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
                
        return embedding_matrix 
        
    def load_fasttext(self, EMBEDDING_FILE):    
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        nb_words = self.vocab_size
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in self.word_to_idx.items():
            if i >= self.vocab_size: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def load_para(self, EMBEDDING_FILE):
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        nb_words = self.vocab_size
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in self.word_to_idx.items():
            if i >= self.vocab_size: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
        return embedding_matrix

    def get_skipgrams(self):
        """Gets all the skipgram pairs needed for doing Lda2Vec.
        These will be the inputs to the model. 

        Note - If a given document ends up having too few tokens in it to compute
        skipgrams, it is thrown away. You can find these thrown away documents in the
        self.purged_docs array.

        Values are stored in a dataframe. The columns are:
        0 - Pivot IDX
        1 - Context IDX
        2 - Unique Doc ID - Takes into consideration the purged docs by not including them.
            Unique Doc ID is what we will use to create the doc_embedding matrix.
        3 - Original Doc ID - Doc ID without considering purged docs. 
        """
        # List to hold skipgrams and associated metadata
        skipgram_data    = []
        # List of indexes (of original texts list from dataframe) where we are throwing the doc out due to short length
        self.purged_docs = []
        # List holding the number of tokens in each document
        self.doc_lengths = []
        # ID to count document IDS
        doc_id_counter = 0
        print("\n---------- Getting Skipgrams ----------")
        for i, t in tqdm(enumerate(self.idx_data)):
            pairs, _ = skipgrams(t,
                                 vocabulary_size = self.vocab_size,
                                 window_size = self.window_size,
                                 shuffle=True,
                                 negative_samples=0)
            
            if len(pairs) > 2:
                for pair in pairs:
                    temp_data = pair
                    # Appends doc ID
                    temp_data.append(doc_id_counter)
                    # Appends document index (index of original texts list from dataframe)
                    temp_data.append(i)
                    skipgram_data.append(temp_data)
                self.doc_lengths.append(len(t))
                doc_id_counter+=1
            else:
                # We purge docs with less than 2 pairs
                self.purged_docs.append(i)

        self.skipgrams_df = pd.DataFrame(skipgram_data)

    def preprocess(self):
        self.tokenize_and_process()
        self.get_supplemental_data()
        self.get_skipgrams()

    def save_data(self, path, embedding_matrix=None):
        """Save all the preprocessed data to a given path. Optionally, you can
        save the embedding matrix in the same path by passing it to the "embedding_matrix" param.

        This embedding matrix should have been created using load_glove, load_para, or load_fasttext functions.
        If not, just make sure the embedding matrix lines up with the values in word_to_idx and that 
        
        Args:
            path (TYPE): Description
            embedding_matrix (None, optional): Description
        """
        if not os.path.exists(path):
            os.makedirs(path)


        # If embedding matrix is passed, save it as long as embedding_matrix.shape[0] == self.vocab_size
        if isinstance(embedding_matrix, type(np.empty(0))):
            assert embedding_matrix.shape[0] == self.vocab_size, "embedding_matrix.shape[0] should match vocab_size - {0} != {1}".format(embedding_matrix.shape[0], self.vocab_size)
            np.save(path+"/embedding_matrix", embedding_matrix)
        else:
            assert embedding_matrix==None, "If you want to save embeddings, they should should be type numpy.ndarray, not {}".format(type(embedding_matrix))
        

        # Save vocabulary dictionaries to file
        idx_to_word_out = open(path + "/idx_to_word.pickle", "wb")
        pickle.dump(self.idx_to_word, idx_to_word_out)
        idx_to_word_out.close()
        word_to_idx_out = open(path + "/word_to_idx.pickle", "wb")
        pickle.dump(self.word_to_idx, word_to_idx_out)
        word_to_idx_out.close()        

        np.save(path+"/doc_lengths", self.doc_lengths)
        np.save(path+"/freqs", self.freqs)
        self.skipgrams_df.to_csv(path+"/skipgrams.txt", sep="\t", index=False, header=None)
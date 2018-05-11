import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from keras.preprocessing.sequence import skipgrams
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Lda2vec_Preprocessor():
    """
    Args:
    path_to_file: Path to a file with sentences separated by a new line (string)
    file_out_path: Path to file that skipgram pairs and doc ids will be stored in (string)
    limit_vocab_size: Set a limit to the number of words in the vocabulary (int)
    compute_skipgrams: Compute skipgrams and save to file_out_path. Turn to False if you already have this file (bool)
    load_glove_embeddings: If you want to load an embedding matrix from glove embeddings, set this to True (bool)
    glove_embeddings_file: Path to GloVe embeddings file. You MUST pass this parameter if load_glove_embeddings is True (string)
    write_every: How many sentences to apply skipgrams to before writing cache to file and clearing cache
    Returns:
        word_to_idx: Dictionary mapping words to indexes (dict)
        idx_to_word: Dictionary mapping indexes of words to actual words (dict)
        T: Keras Tokenizer Object (Keras Tokenizer)
        freqs: ordered list of word frequencies, ordered by idx (Numpy array)


    Optionally Creates:
        embed_matrix: Pretrained embedding matrix from Glove embeddings
    """

    def __init__(self,
                 path_to_file,
                 file_out_path,
                 limit_vocab_size=10000,
                 compute_skipgrams=True,
                 window_size=2,
                 load_pretrained_embeddings=False,
                 glove_embeddings_file=None,
                 init_oov_randomly=True,
                 write_every=10000,
                 extract_data=False):
        self.path_to_file = path_to_file
        self.file_out_path = file_out_path
        self.vocab_size = limit_vocab_size
        self.window_size = window_size
        self.compute_skipgrams = compute_skipgrams
        self.load_pretrained_embeddings = load_pretrained_embeddings
        self.glove_embeddings_file = glove_embeddings_file
        self.init_oov_randomly = init_oov_randomly
        self.write_every = write_every

        if compute_skipgrams == True:
            assert not (os.path.exists(self.file_out_path)), ("You should change the name of your file out path or change compute_skipgrams to false if your file out path already exists.")
        # Make sure embeddings file passed in if you want to load pretrained embeddings from glove
        if self.load_pretrained_embeddings == True:
            assert (self.glove_embeddings_file != None), ("If load_glove_embeddings is true, you must pass a file to glove_embeddings_file parameter")

        # Read data into dataframe
        df = pd.read_csv(path_to_file, sep="\t", header=None)

        # Extract the text from the dataframe
        self.text = df[0].values.tolist()

        # Reduce memory by deleting dataframe
        del df

        # Tokenize and process the data
        self.tokenize_and_process()

        if compute_skipgrams:
            self.concat_skipgrams()

        if extract_data == False:
            del self.data
        del self.doc_ids

        if self.load_pretrained_embeddings == True:
            self.load_glove_as_dict()
            self.get_embed_matrix_from_glove()
            # return word_to_idx, idx_to_word, T, freqs, embed_matrix
        # return word_to_idx, idx_to_word, T, freqs

    def tokenize_and_process(self):
        # Will hold clean text
        text_clean = []
        # Puts together list of words we don't want to include
        stop = stopwords.words('english') + list(string.punctuation)

        # Iterate through sentences
        for t in self.text:
            text_clean.append(" ".join([i for i in word_tokenize(t.lower()) if i not in stop]))

        # Instantiate tokenizer, pass it vocab size
        self.T = Tokenizer(num_words=self.vocab_size)

        # Fit the tokenizer with the texts
        self.T.fit_on_texts(text_clean)

        # Turns our input text into sequences of index numbers
        self.data = self.T.texts_to_sequences(text_clean)

        # Subtract 1 from each index
        self.data = [[col+1 for col in row] for row in self.data]

        # Delete to reduce memory
        del text_clean

        # Extract word counts (frequency of words) from tokenizer object
        word_counts = self.T.word_counts

        # If we find that the given amount of unique words is less than set vocab size, resize the vocab
        if len(word_counts) < self.vocab_size:
            self.vocab_size = len(word_counts)

        # Convert these frequencies to pandas dataframe
        df_freqs = pd.DataFrame.from_records(word_counts, index=["Freqs"])
        # Transpose df_freqs, as it is in the wrong format
        df_freqs = df_freqs.T
        # Now, we order this dataframe by decending frequency of words
        df_freqs = df_freqs.sort_values(["Freqs"], ascending=False)
        # Extract these frequencies to a numpy array
        self.freqs = df_freqs.values.flatten().tolist()[:self.vocab_size]

        # Creates unique document Ids
        self.doc_ids = np.arange(len(self.data))
        # Get the number of unique documents
        self.num_unique_documents = self.doc_ids.max()

        # Gets our two translation dictionaries
        #self.word_to_idx = self.T.word_index
        self.word_to_idx = {k: v-1 for k, v in self.T.word_index.items()}

        # Flip the dictionary for word_to_idx
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

    def concat_skipgrams(self):
        """
        Args:
        text: cleaned text from preprocess function
        doc_ids: list of unique id numbers for each document
        vocab_size: size of our vocabulary
        write_every: amount of sentences to process before appending to file_out
        window_size: skipgram window size (2 on left of pivot word, 2 on right of pivot word)
        file_out_path: path to save skipgram pairs (and doc ids) to file. Should be .csv or .txt

        """
        data = []

        # Loop through sentences
        for i, t in enumerate(self.data):
            pairs, _ = skipgrams(t,
                                 vocabulary_size=self.vocab_size,
                                 window_size=self.window_size,
                                 shuffle=True,
                                 negative_samples=0)
            for pair in pairs:
                temp_data = pair
                temp_data.append(self.doc_ids[i])
                data.append(temp_data)
            if i // self.write_every:
                temp_df = pd.DataFrame(data)
                temp_df.to_csv(file_out_path, sep="\t",
                               index=False, header=None, mode="a")
                del temp_df
                data = []
        temp_df = pd.DataFrame(data)
        temp_df.to_csv(self.file_out_path, sep="\t",
                       index=False, header=None, mode="a")
        del temp_df

    def load_glove_as_dict(self):
        '''
        This function will read glove embeddings
        into a dictionary, allowing you to extract
        only the pretrained embeddings for the words
        in your corpus. This function allows for
        variable length embedding sizes, as long as
        they are in the same format...first row element
        must be word, rest are embedding values
        ----------------------------------------------
        Sets:
        glove_dict - Dictionary where keys are words and values
                    are pretrained embeddings
        embedding_dim - The dimension of each word embedding (embedding size) 
        '''
        self.glove_dict = {}
        file = open(self.glove_embeddings_file, 'r', encoding="latin")
        # This will allow us to check the embedding size of the pretrained embeddings
        check_embed_dim = True
        for line in file.readlines():
            # Extract the row and split into a list
            row = line.strip().split(' ')
            if check_embed_dim == True:
                self.embedding_dim = len(row[1:])
                check_embed_dim = False
            try:
                # The first element in the row is the word, rest are the embedding
                self.glove_dict[row[0]] = np.asarray(row[1:], dtype='float32')
            except Exception as e:
                print(e)
                print(row[0])
                print(row[1:])
                print(type(row[1:]))
        # Close file
        file.close()

    def get_embed_matrix_from_glove(self):
        '''
        This function will take the word_to_idx dictionary
        that was gathered from preprocessing our own corpus,
        and will extract all pretrained embeddings from the glove
        embeddings to form a pretrained embedding matrix. 

        Additionally, we have the option to randomly initialize
        word embeddings for words that are found in our corpus, but not
        the glove corpus.
        '''
        # Create empty embedding matrix with zeros.
        # We instantiate it to be of shape [vocabulary size, embedding size]
        self.embed_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        print("number of keys in idx_to_word", len(self.idx_to_word.keys()))
        for idx in range(len(self.idx_to_word.keys())):
            # Get the word for the given index
            word = self.idx_to_word[idx]
            # Checks to see if a given word is in the glove vocabulary
            word_vector = self.glove_dict.get(word)

            # Sets embedding vector for the given index
            if word_vector is not None:
                self.embed_matrix[idx] = word_vector
            # Sets a random uniform vector for the given index
            elif self.init_oov_randomly == True:
                self.embed_matrix[idx] = np.random.uniform(-1, 1, self.embedding_dim)

    def get_training_data(self):
        # Read in processed data
        df = pd.read_csv(self.file_out_path, sep="\t", header=None)
        pivot_words = df[0].values
        target_words = df[1].values
        doc_ids = df[2].values
        # Shuffle data
        pivot_words, target_words, doc_ids = shuffle(pivot_words, target_words, doc_ids, random_state=10)
        return pivot_words, target_words, doc_ids

    # Taken directly from https://stackoverflow.com/questions/17322273/store-a-dictionary-in-a-file-for-later-retrieval
    def SaveDictionary(self, dictionary, File):
        with open(File, "wb") as myFile:
            pickle.dump(dictionary, myFile)
            myFile.close()

    def LoadDictionary(self, File):
        with open(File, "rb") as myFile:
            dict = pickle.load(myFile)
            myFile.close()
            return dict

    def translate_seq_to_text(self, seqs):
        words = []
        for seq in seqs:
            seq = np.trim_zeros(seq)
            words.append(" ".join([self.idx_to_word[idx] for idx in seq]))
        return words

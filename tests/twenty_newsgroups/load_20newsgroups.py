import pandas as pd
import numpy as np
from lda2vec.nlppipe import Preprocessor

# Data directory
data_dir ="data"
# Where to save preprocessed data
clean_data_dir = "data/clean_data"
# Name of input file. Should be inside of data_dir
input_file = "20_newsgroups.txt"

# Read in data file
df = pd.read_csv(data_dir+"/"+input_file, sep="\t")

# Initialize a preprocessor
P = Preprocessor(df, "texts", max_features=30000, min_count=30)

# Run the preprocessing on your dataframe
P.preprocess()

# Load embedding matrix from file path
embedding_matrix = P.load_glove("glove_embeddings/glove.6B.300d.txt")

# # # Save data to data_dir
P.save_data(clean_data_dir, embedding_matrix=embedding_matrix)

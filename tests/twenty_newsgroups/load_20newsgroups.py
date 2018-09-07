import pandas as pd
from lda2vec import utils

# Data directory
data_dir ="data"
# Subdirectory of data_dir to store preprocessed data
run_name = "my_run"
# Name of input file. Should be inside of data_dir
input_file = "20_newsgroups.txt"

# Read in data file
df = pd.read_csv(data_dir+"/"+input_file, sep="\t")

# Extract Texts
texts = df.texts.values.tolist()

# Add in words to filter out
bad = set(["ax>", '`@("', '---', '===', '^^^', "AX>", "GIZ"])

# Run the preprocessing. Preprocessed data will be found under "data_dir/run_name/"
utils.run_preprocessing(texts, data_dir, run_name, bad=bad, max_length=10000)

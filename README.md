# recurrent-rebels

MLX Institute, Recurrent Rebels group

## 1. Exploratory Data Analysis (EDA)

a. look for cues

- Rasched/Nick, exploring for features in SQL/Pandas

user features:
karma
no of submitted posts
max/min/mean upvotes per post 
post per year 
total posts 
descendentans 
deepest sub comment level? 
no of comments / sub comments 
how many authored 
how many comments 
time they have been a user 
avg number of people engaging on their post (same as avg number of comments)


- How to embed users? use something like Node2Vec or other graph embeddings

b. understand the distributions

c. data quality issues

- Anton, notebook connecting to SQL

### Downloading data from the HackerNews DB

`download.py` will download selected comments and titles from the database (136 MB total) as well as the complete set of posts (800 MB). 

`tokenizer.py` will tokenize comments and titles into `hn_corpus.txt`.

### Directories

- `data` holds raw data files, including downloads from SQL
- `embeddings` and `skipgram_models` hold models built with word2vec

## 2. Build prediction models

To run the word2vec code, run python word2vec.py from the command line. Or use the notebook word2vec.ipynb, which wraps it, on Google Colab. This produces embeddings in the filename specified as the argument. Specify `--corpus` for the corpus and `--model` for the output file.

Once the model is built, you can test it with `python tester.py <word> <model-file>`. Or even a bunch of words: `python word2vec_test.py <model-file>`

We can evaluate models side by side with `python word2vec_eval.py model1 model2`

c. finetune embeddings

d. regression task
### Prediction Model Features

- title: words -> dense -> mean (param: emb_dim), length?
- domain: learned dense embedding (param: emb_dim)
- user: several scalar features
- time: hour, day of week (each 2 scalars), 

## 3. Deploy a predictor service

a. grab a server on Computa

b. run Docker container

c. respond to live requests

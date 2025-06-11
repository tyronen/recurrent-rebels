# recurrent-rebels

MLX Institute, Recurrent Rebels group

## How to run this project

Install requirements: Run `pip install -r requirements.txt`

## 1. Exploratory Data Analysis (EDA)

a. look for cues

- Rasched/Nick, exploring for features in SQL/Pandas

# 🔎 Feature Brainstorming for Hacker News Upvote Prediction

## ✅ User Features

- Karma  
- Number of submitted posts (total posts)  
- Max upvotes per post  
- Min upvotes per post  
- Mean upvotes per post  
- Posts per year  
- Total posts  
- Descendants (total number of comments under the post)  
- Deepest sub comment level  
- Number of comments / sub comments  
- Mean number of comments per post  
- Total number of comments  
- Account age (time they have been a user)  
- Average number of people engaging on their post (same as average comments per post?)

## 🕒 Temporal Features

- Post time of day (hour bucket: morning, afternoon, evening, night)
- Post weekday/weekend indicator
- Account age at post time (how old the account was when posting)
- Post within first N minutes of daily HN activity cycle
- Days since previous post

## 🔄 User Interaction Diversity

- Number of distinct people commenting on user’s posts
- Number of unique threads user participated in (non-own posts)
- Fraction of posts that got at least 1 upvote (success rate)

## 📝 Post Content Features

- Title length (word count, character count)
- Average word length in title
- Presence of question mark (`?`)
- Presence of exclamation mark (`!`)
- Sentiment score of title
- Embedding of title (TF-IDF, Word2Vec, BERT etc.)
- Title language (non-English titles tend to perform worse): can do this as binary

## 🔗 URL Features

- Domain popularity (e.g., github.com, nytimes.com, etc.)
- External domain reputation (popular domains may have higher base upvotes)
- Is submission linking to user's own site (self-promotion)

## 👥 Community-specific Features

- Submission category/topic (AI, Crypto, Programming, Startups, etc.)
- Average performance of user's previous posts in same category
- Number of reposts of the same URL

## ⚡ Meta Engagement Features

- Average time to first comment
- Time to 10 comments (velocity indicator)
- Fraction of posts receiving moderator flag
- Upvote velocity (upvotes per minute/hour after posting)
- Time taken to hit score thresholds (10, 50, 100 upvotes)

## 💡 Additional Advanced Features

- Network centrality (user’s interaction graph on HN)
- User influence score (e.g., followers or mentions elsewhere)

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

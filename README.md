# Test-Mining-Elon-musk-and-Amazon-review-data
For Text Mining assignment    ONE: 1) Perform sentimental analysis on the Elon-musk tweets (Exlon-musk.csv)   TWO: 1) Extract reviews of any product from ecommerce website like amazon 2) Perform emotion mining
import nltk
nltk.download('vader_lexicon')
import nltk
nltk.download('punkt')
import pandas as pd
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from nltk import ngrams
from collections import Counter

# Download the NLTK 'punkt' resource
import nltk
nltk.download('punkt')

# Your existing code for text mining
# Load the dataset from the CSV file with the specified encoding
data = pd.read_csv('Elon_musk.csv', encoding='latin1')

# Part One: Sentiment Analysis on Elon Musk's Tweets
# Perform sentiment analysis on each tweet
sentiments = []
for tweet in data['Text']:
    blob = TextBlob(tweet)
    sentiment = blob.sentiment.polarity
    sentiments.append(sentiment)

# Add sentiment scores to the dataset
data['Sentiment'] = sentiments

# Visualize sentiment distribution with a histogram
plt.figure(figsize=(8, 6))
plt.hist(data['Sentiment'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Sentiment Analysis of Elon Musk Tweets')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()
import nltk
nltk.download('averaged_perceptron_tagger')
pip install requests beautifulsoup4 nltk textblob wordcloud matplotlib seaborn
# Part Two: Extract Reviews from an E-commerce Website (Amazon) and Perform Emotion Mining

# Extract reviews from an Amazon product page
url = 'https://www.amazon.com/Kindle-Paperwhite-Waterproof-Storage-Special/dp/B075MWNNWQ'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

reviews = soup.find_all('div', class_='review')
review_texts = [review.find('span', class_='review-text').get_text() for review in reviews]

# Perform emotion mining on the extracted reviews
sia = SentimentIntensityAnalyzer()
sentiments = []

for review_text in review_texts:
    sentiment = sia.polarity_scores(review_text)
    sentiments.append(sentiment['compound'])  # Using compound score for emotion mining

# Create a DataFrame to store review texts and sentiment scores
data_reviews = pd.DataFrame({'Review': review_texts, 'Sentiment': sentiments})

# Visualize sentiment distribution of reviews with a histogram
plt.figure(figsize=(8, 6))
sns.histplot(data_reviews['Sentiment'], bins=20, kde=True)
plt.title('Emotion Mining of Amazon Product Reviews')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Visualize relationship between sentiment and review length with a scatterplot
review_lengths = [len(review.split()) for review in review_texts]
plt.figure(figsize=(8, 6))
sns.scatterplot(x=review_lengths, y=data_reviews['Sentiment'])
plt.title('Relationship between Review Length and Sentiment')
plt.xlabel('Review Length')
plt.ylabel('Sentiment Score')
plt.show()

# Named Entity Recognition (NER) using TextBlob
ner_tags = []
for tweet in data['Text']:
    blob = TextBlob(tweet)
    ner_tags.append(blob.tags)

# Flatten the list of lists
ner_tags_flat = [item for sublist in ner_tags for item in sublist]

# Filter out named entities
named_entities = [word for word, tag in ner_tags_flat if tag == 'NNP']

# Word Cloud Visualization for Named Entities
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(named_entities))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Named Entity Word Cloud')
plt.axis('off')
plt.show()

# N-gram and Bi-gram Analysis
# Convert text data to lowercase
text_data = ' '.join(data['Text']).lower()

# Tokenize the text
tokens = text_data.split()

from nltk.util import ngrams
from collections import Counter

# Create n-grams and bi-grams
n_grams_data = ngrams(tokens, 3)  # Change the number to adjust n-grams
n_grams_freq = Counter(n_grams_data)

# Extract the n-grams and their frequencies
ngram, freq = zip(*n_grams_freq.most_common(10))

# Extract n-grams from the tuple of tuples
ngram = [' '.join(gram) for gram in ngram]

# Now, try plotting the bar chart again
plt.bar(ngram, freq)
plt.title('Top 10 Most Common N-grams')
plt.xlabel('N-grams')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


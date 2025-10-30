import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download necessary NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text = "This is an example document, showing off the stop words filtration and stemming process."

# Tokenization
tokens = nltk.word_tokenize(text)

# Convert to lower case and remove punctuation
tokens = [word.lower() for word in tokens if word.isalpha()]

# Stopword removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

# Output results
print("Original Tokens:", tokens)
print("After Stopword Removal:", filtered_tokens)
print("After Stemming:", stemmed_tokens)

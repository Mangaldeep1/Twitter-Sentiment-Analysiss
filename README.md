# Twitter-Sentiment-Analysiss
# Sentiment Analysis with NLTK + Scikit-learn
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Download NLTK data (run once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Example dataset (replace with real Twitter data)
data = {
    'tweet': [
        "I love this phone, it's amazing!",
        "This is the worst movie ever",
        "I am feeling okay today",
        "Such a beautiful day!",
        "I hate waiting in long lines"
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
}
df = pd.DataFrame(data)

# Text preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)      # remove mentions
    text = re.sub(r"#\w+", "", text)      # remove hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['clean_tweet'] = df['tweet'].apply(clean_text)

# Convert text to features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_tweet'])
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

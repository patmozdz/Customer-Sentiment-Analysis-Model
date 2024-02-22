import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load spaCy's English-language models
nlp = spacy.load("en_core_web_sm")


# Load JSON data into a pandas DataFrame
def load_data(filename):
    return pd.read_json(filename, lines=True)


# Preprocess text data
def preprocess_texts(texts):
    preprocessed_texts = []
    for doc in nlp.pipe(texts, disable=["parser", "ner"], batch_size=50, n_process=-1):
        preprocessed_texts.append(" ".join([token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]))
    return preprocessed_texts


# Load data
df_reviews = load_data('amazon_reviews_data.json')

# Preprocess reviews
df_reviews['processed_reviewText'] = preprocess_texts(df_reviews['reviewText'])

# Convert overall ratings into binary labels (4 or 5 stars is positive, 1, 2, or 3 stars is negative)
df_reviews['label'] = (df_reviews['overall'] >= 4).astype(int)

# Feature extraction with Bag of Words
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df_reviews['processed_reviewText']).toarray()
y = df_reviews['label']

# Splitting the dataset into the training set and test set (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

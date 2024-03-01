import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm.auto import tqdm
import joblib
from sqlalchemy import create_engine, Table, MetaData, Column, String, DateTime, Integer, Float
from datetime import datetime

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")


# Load JSON data into a pandas DataFrame
def load_data(filename):
    return pd.read_json(filename, lines=True)


# Preprocess text data
def preprocess_texts(texts):
    preprocessed_texts = []
    for doc in tqdm(nlp.pipe(texts, disable=["parser", "ner"], batch_size=50, n_process=1), total=len(texts), desc="Preprocessing Text"):
        preprocessed_texts.append(" ".join([token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]))
    return preprocessed_texts


# Function to insert model metadata into the database using SQLAlchemy
def save_model_metadata(engine, model_name, model_path, vectorizer_path, version):
    metadata = MetaData()
    models = Table('models', metadata,
                   Column('model_name', String),
                   Column('model_path', String),
                   Column('vectorizer_path', String),
                   Column('created_at', DateTime),
                   Column('version', String))
    insert_stmt = models.insert().values(
        model_name=model_name,
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        created_at=datetime.now(),
        version=version
    )
    with engine.connect() as connection:
        connection.execute(insert_stmt)
        connection.commit()


# Function to insert model metrics into the database using SQLAlchemy
def save_metrics(engine, accuracy, precision, recall, f1_score):
    metadata = MetaData()
    metrics_table = Table('metrics', metadata,
                   Column('accuracy', Float),
                   Column('precision', Float),
                   Column('recall', Float),
                   Column('f1_score', Float),
                   Column('created_at', DateTime))
    insert_stmt = metrics_table.insert().values(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),  # En
        f1_score=float(f1_score),
        created_at=datetime.now()
    )
    with engine.connect() as connection:
        connection.execute(insert_stmt)
        connection.commit()


def save_statistics(engine, positive_test, negative_test, positive_pred, negative_pred):
    metadata = MetaData()
    stats_table = Table('statistics', metadata,
                   Column('positive_test', Integer),
                   Column('negative_test', Integer),
                   Column('positive_pred', Integer),
                   Column('negative_pred', Integer),
                   Column('created_at', DateTime))
    insert_stmt = stats_table.insert().values(
        positive_test=positive_test,
        negative_test=negative_test,
        positive_pred=positive_pred,
        negative_pred=negative_pred,
        created_at=datetime.now()
    )
    with engine.connect() as connection:
        connection.execute(insert_stmt)
        connection.commit()


def calculate_failure_statistics(y_test, predictions, ratings):
    failure_stats = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Loop through all the predictions and actual ratings
    for actual_label, predicted_label, rating in zip(y_test, predictions, ratings):
        # If the prediction is incorrect and the actual rating was negative (considering 1-3 as negative, 4-5 as positive)
        if predicted_label != actual_label and rating in failure_stats:
            failure_stats[rating] += 1

    return failure_stats


def save_failure_statistics(engine, failure_stats):
    metadata = MetaData()
    failure_stats_table = Table('failure_statistics', metadata, autoload_with=engine)

    with engine.connect() as connection:
        for rating, incorrect_predictions in failure_stats.items():
            insert_stmt = failure_stats_table.insert().values(
                rating=rating,
                incorrect_predictions=incorrect_predictions,
                created_at=datetime.now()
            )
            connection.execute(insert_stmt)
        connection.commit()


def main():
    # SQLAlchemy engine for SQLite
    engine = create_engine('sqlite:///sentiment_analysis_db.sqlite')

    # Load data
    df_reviews = load_data('amazon_reviews_data.json')

    # Preprocess reviews
    df_reviews['processed_reviewText'] = preprocess_texts(df_reviews['reviewText'])

    # Convert overall ratings into binary labels
    df_reviews['label'] = (df_reviews['overall'] >= 4).astype(int)

    # Feature extraction with Bag of Words
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df_reviews['processed_reviewText']).toarray()
    y = df_reviews['label']

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate performance
    print("Accuracy:", accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    print("\nClassification Report:\n", classification_report(y_test, predictions))

    # Compute statistics
    positive_test = sum(y_test)
    negative_test = len(y_test) - positive_test
    positive_pred = int(sum(predictions))
    negative_pred = int(len(predictions) - positive_pred)

    # Compute failure statistics
    failure_stats = calculate_failure_statistics(y_test, predictions, df_reviews['overall'][y_test.index])

    # Save the model and vectorizer
    model_path = "random_forest_model.joblib"
    vectorizer_path = "vectorizer.joblib"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    save_metrics(engine, accuracy_score(y_test, predictions), precision, recall, f1_score)
    save_statistics(engine, positive_test, negative_test, positive_pred, negative_pred)
    save_failure_statistics(engine, failure_stats)
    save_model_metadata(engine, "RandomForestClassifier", model_path, "vectorizer.joblib", "v1.0")


if __name__ == "__main__":
    main()

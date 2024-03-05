import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk
import joblib
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from pre_process_and_train import preprocess_texts
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# To convert this script to exe: pyinstaller --onefile --icon=app_icon.ico tkinter_app.py

# Determine if we're running as a script or frozen exe (pyinstaller)
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)

model_path = os.path.join(application_path, 'random_forest_model.joblib')
database_path = os.path.join(application_path, 'sentiment_analysis_db.sqlite')


# Check and setup database if needed
def setup_database():
    if not os.path.exists(database_path):
        print("Database not found. Setting up database...")
        subprocess.run(["poetry", "run", "python", "setup_database.py"])
    else:
        print("Database already set up.")


# Check if the model exists, and train if it doesn't
def train_model_if_needed():
    if not os.path.exists(model_path):
        print("Model not found. Training model...")
        subprocess.run(["poetry", "run", "python", "pre_process_and_train.py"])
    else:
        print("Model already trained.")


# Load the trained model
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        tk.messagebox.showerror("Error", "Model file not found.")
        exit()


def fetch_statistics():
    engine = create_engine('sqlite:///sentiment_analysis_db.sqlite')
    with Session(engine) as session:
        result = session.execute(text("SELECT * FROM statistics ORDER BY created_at DESC LIMIT 1"))
        stats = result.fetchone()
    return dict(stats._mapping)


def show_statistics():
    stats = fetch_statistics()
    if stats is not None:
        labels = ['Real Positives', 'Real Negatives', 'Positive Predictions', 'Negative Predictions']
        values = [stats['positive_test'], stats['negative_test'], stats['positive_pred'], stats['negative_pred']]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.xlabel('Statistics')
        plt.ylabel('Count')
        plt.title('Test Data Real Ratings vs Predictions')
        plt.show()
    else:
        tk.messagebox.showinfo("Statistics", "No statistics available.")


def fetch_metrics():
    engine = create_engine('sqlite:///sentiment_analysis_db.sqlite')
    with Session(engine) as session:
        result = session.execute(text("SELECT * FROM metrics ORDER BY created_at DESC LIMIT 1"))
        metrics = result.fetchone()
    return dict(metrics._mapping)


def show_metrics():
    metrics = fetch_metrics()
    if metrics is not None:
        labels = np.array(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
        stats = np.array([metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]) * 100

        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        stats = np.concatenate((stats, [stats[0]]))  # Complete the loop
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_ylim(80, 85)  # Adjust the scale to your metrics' range

        ax.fill(angles, stats, color='green', alpha=0.25)
        ax.plot(angles, stats, color='green', linewidth=2)

        for angle, stat, label in zip(angles, stats, labels):
            if angle in (0, np.pi):
                alignment = 'center'
            elif 0 < angle < np.pi:
                alignment = 'left'
            else:
                alignment = 'right'

            ax.text(angle, stat + 0.5, f"{stat:.2f}", ha=alignment, va='center')  # Adjust text position

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        plt.title('Model Performance Metrics', size=20, color='green', y=1.1)
        plt.subplots_adjust(top=0.85)
        plt.show()
    else:
        tk.messagebox.showinfo("Metrics", "No metrics available.")


def fetch_failure_statistics():
    engine = create_engine('sqlite:///sentiment_analysis_db.sqlite')
    with Session(engine) as session:
        result = session.execute(text("SELECT rating, incorrect_predictions FROM failure_statistics"))
        failure_stats = result.fetchall()
        failure_stats_dict = {row.rating: row.incorrect_predictions for row in failure_stats}
    return failure_stats_dict


def show_failure_statistics():
    failure_stats = fetch_failure_statistics()
    if failure_stats:
        ratings = [f'{rating} stars' for rating in failure_stats.keys()]
        incorrect_predictions = list(failure_stats.values())

        plt.figure(figsize=(8, 8))
        plt.pie(incorrect_predictions, labels=ratings, autopct='%1.1f%%', startangle=140)
        plt.title('Proportion of Incorrect Predictions by Rating')
        plt.show()
    else:
        tk.messagebox.showinfo("Failure Statistics", "No failure statistics available.")


def save_review(review_text, rating):
    engine = create_engine('sqlite:///sentiment_analysis_db.sqlite')
    with Session(engine) as session:
        # SQL statement to insert the new review
        insert_statement = text(
            "INSERT INTO reviews (review_text, rating, created_at) VALUES (:review_text, :rating, :created_at)"
        )
        # Executing the insert statement with parameters
        session.execute(insert_statement, {'review_text': review_text, 'rating': rating, 'created_at': datetime.now()})
        session.commit()


def analyze_review():
    review_text = review_entry.get("1.0", "end-1c")  # Get text from text entry
    if review_text == "":
        output_box.delete('1.0', tk.END)  # Clear the output box before showing new output
        output_box.insert(tk.END, "Please enter a valid review.")  # Show an error message
        return

    output_box.delete('1.0', tk.END)  # Clear the output box
    output_box.insert(tk.END, "...")  # Insert placeholder text (this is done because sometimes it's confusing if you hit "Analyze Review" and nothing happens)

    # Schedule display_analysis_result to run after 500ms
    app.after(500, lambda: display_analysis_result(review_text))


def display_analysis_result(review_text):
    preprocessed_review = preprocess_texts([review_text])  # Preprocess the review text
    vectorized_review = vectorizer.transform(preprocessed_review).toarray()
    prediction = model.predict(vectorized_review)

    output_box.delete('1.0', tk.END)  # Clear the output box before showing new output
    if prediction[0] == 1:
        output_box.insert(tk.END, "This review is Positive.")
        save_review(review_text, 1)  # Save the review with a positive rating
    else:
        output_box.insert(tk.END, "This review is Negative.")
        save_review(review_text, 0)  # Save the review with a negative rating


# GUI setup
def setup_gui():
    global app
    app = tk.Tk()
    app.title("Review Sentiment Analysis")
    app.geometry('550x325')

    review_label = ttk.Label(app, text="Enter a review:")
    review_label.pack()

    global review_entry
    review_entry = tk.Text(app, height=10, width=65)
    review_entry.pack()

    # Buttons frame for horizontal layout
    buttons_frame = tk.Frame(app)
    buttons_frame.pack(pady=10)

    analyze_button = ttk.Button(buttons_frame, text="Analyze Review", command=analyze_review)
    analyze_button.pack(side=tk.LEFT, padx=5)  # Pack buttons side by side

    stats_button = ttk.Button(buttons_frame, text="Show Prediction Statistics", command=show_statistics)
    stats_button.pack(side=tk.LEFT, padx=5)  # Add padding between buttons

    metrics_button = ttk.Button(buttons_frame, text="Show Model Metrics", command=show_metrics)
    metrics_button.pack(side=tk.LEFT, padx=5)

    failure_stats_button = ttk.Button(buttons_frame, text="Show Failure Statistics", command=show_failure_statistics)
    failure_stats_button.pack(side=tk.LEFT, padx=5)

    # Output box for displaying the analysis result
    global output_box
    output_box = tk.Text(app, height=4, width=65)
    output_box.pack(pady=10)  # Add some padding for better spacing

    app.mainloop()


if __name__ == '__main__':
    setup_database()  # Setup database if needed
    train_model_if_needed()  # Check and potentially train model
    model = load_model()  # Load the model for the application
    vectorizer = joblib.load('vectorizer.joblib')  # Load the vectorizer
    setup_gui()  # Setup and run the Tkinter GUI

# Spam/Ham SMS Classifier

A machine learning project that classifies SMS messages as spam or ham (legitimate) using natural language processing and ensemble learning techniques.

## Overview

This project implements a comprehensive SMS/email spam detection system using various machine learning algorithms and NLP techniques. The final model achieves high accuracy through ensemble voting of multiple classifiers.

## Features

- **Data Cleaning & Preprocessing**: Handles missing values, duplicates, and text normalization
- **Exploratory Data Analysis**: Visualizations including word clouds, correlation heatmaps, and distribution plots
- **Feature Engineering**:
  - Character, word, and sentence count analysis
  - TF-IDF vectorization with 3000 max features
  - Text transformation (lowercase, tokenization, stopword removal, stemming)
- **Multiple ML Algorithms Tested**:
  - Naive Bayes (Gaussian, Multinomial, Bernoulli)
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - AdaBoost, Gradient Boosting, XGBoost
  - Bagging and Extra Trees
- **Ensemble Learning**: VotingClassifier combining best performing models
- **Model Persistence**: Trained models saved using pickle

## Dataset

The project uses the SMS Spam Collection dataset containing labeled SMS messages (spam/ham).

## Technologies Used

- **Python 3.x**
- **Libraries**:
  - NumPy & Pandas - Data manipulation
  - Scikit-learn - Machine learning algorithms
  - NLTK - Natural language processing
  - Matplotlib & Seaborn - Data visualization
  - WordCloud - Text visualization
  - XGBoost - Gradient boosting

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ayushhkr/spam-ham-classifierrr.git
cd spam-ham-classifierrr
```

2. Install required packages:

```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn wordcloud xgboost
```

3. Download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

1. Open the Jupyter notebook:

```bash
jupyter notebook spam_ham_classifier.ipynb
```

2. Run all cells to train the model

3. Use the saved models for predictions:

```python
import pickle

# Load the models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Make prediction on a new message
message = "Congratulations! You've won a free iPhone"
transformed_msg = tfidf.transform([message])
prediction = model.predict(transformed_msg)[0]
print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")
```

**Quick Prediction Example (from notebook):**

```python
# Example: Detect spam in a message
msg = "Congratulations! You've won a $1000 gift card. Click here to claim now!"
transformed_msg = tfidf.transform([msg])
prediction = mnb.predict(transformed_msg)[0]
# Output: Spam ✓
```

## Model Performance

The final ensemble model (VotingClassifier with SVC, MultinomialNB, and ExtraTreesClassifier) achieves:

- **Accuracy**: 97.87% on test set
- **Precision**: 100% - Perfect spam detection with zero false positives
- **Best Individual Models**:
  - MultinomialNB: 96.62% accuracy
  - Extra Trees Classifier: 97.49% accuracy
  - Support Vector Machine: High performance with sigmoid kernel
- Performance metrics tracked and compared across 11+ classification algorithms

## Project Structure

```
spam-ham-classifierrr/
├── spam_ham_classifier.ipynb   # Main notebook with full analysis
├── vectorizer.pkl               # Saved TF-IDF vectorizer
├── model.pkl                    # Saved trained model
├── model_spam_ham.pkl          # Alternative model pickle
├── transform_text.pkl          # Text transformation utilities
└── README.md                   # Project documentation
```

## Key Steps

1. **Data Loading & Cleaning**: Import dataset and handle missing/duplicate values
2. **EDA**: Analyze message lengths, word frequencies, and class distributions
3. **Text Preprocessing**: Tokenization, lowercasing, stopword removal, stemming
4. **Feature Extraction**: TF-IDF vectorization
5. **Model Training**: Train multiple classifiers and compare performance
6. **Ensemble Method**: Combine top models using VotingClassifier
7. **Evaluation**: Assess accuracy, precision, and create confusion matrices
8. **Model Saving**: Serialize models for deployment

## Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## License

This project is open source and available for educational purposes.

## Author

[Ayush Kumar](https://github.com/ayushhkr)

## Acknowledgments

- SMS Spam Collection Dataset
- Scikit-learn documentation
- NLTK community

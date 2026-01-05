# Fake News Classification using Classical NLP

A binary text classification system that distinguishes fake news from factual reporting using classical NLP techniques and machine learning.

## Overview

This project implements fake news detection using:
- **Preprocessing**: Tokenization, stemming, and stopword removal
- **Feature Engineering**: TF-IDF vectorization (5000 features)
- **Models**: Logistic Regression and Linear SVM
- **Evaluation**: Standard classification metrics (accuracy, precision, recall, F1)

## Dataset

The dataset contains 6,335 news articles with the following columns:
- `title`: Article headline
- `text`: Full article content
- `date`: Publication date
- `fake_or_factual`: Label (Fake News or Factual News)

**Class Distribution**: Roughly balanced between fake and factual news

## Methodology

### 1. Data Preprocessing
- Convert text to lowercase
- Remove special characters and digits
- Tokenize into words
- Remove stopwords
- Apply Porter stemming

### 2. Feature Extraction
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Maximum 5000 features with document frequency constraints
- Sparse matrix representation

### 3. Model Training
Two models are trained and compared:

| Model | Type | Algorithm |
|-------|------|-----------|
| Logistic Regression | Linear Classifier | Probabilistic |
| Linear SVM | Support Vector Machine | Hinge loss with SGD |

### 4. Evaluation
Models are evaluated on an 80/20 train-test split with:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Results

Both models achieve strong performance on the test set:

| Metric | Logistic Regression | Linear SVM |
|--------|-------------------|-----------|
| Accuracy | ~0.93 | ~0.92 |
| Precision | ~0.92 | ~0.91 |
| Recall | ~0.94 | ~0.93 |
| F1-Score | ~0.93 | ~0.92 |

The models successfully identify linguistic patterns that distinguish fake from factual news.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fake-news-classification-nlp.git
cd fake-news-classification-nlp

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

```bash
# Run the Jupyter notebook
jupyter notebook notebooks/fake_news_classification.ipynb
```

The notebook includes:
- Data loading and exploration
- Text preprocessing pipeline
- Feature extraction
- Model training and evaluation
- Performance visualization

## Key Findings

1. **Simple Models Work Well**: Classical ML outperforms complex alternatives when proper feature engineering is applied
2. **Text Patterns Matter**: Fake news exhibits distinct vocabulary and linguistic patterns
3. **Balanced Approach**: Using TF-IDF with appropriate thresholds prevents overfitting
4. **Interpretability**: Unlike deep learning, feature importance can be extracted directly

## Tech Stack

- **Data Processing**: Pandas, NumPy
- **NLP**: NLTK
- **Feature Extraction**: Scikit-learn (TfidfVectorizer)
- **Modeling**: Scikit-learn (LogisticRegression, SGDClassifier)
- **Visualization**: Matplotlib, Seaborn

## Limitations & Future Work

### Current Limitations
- Limited to English text
- Binary classification only
- Fixed preprocessing pipeline
- No real-time predictions

### Future Improvements
- Hyperparameter tuning with cross-validation
- Ensemble methods (voting, stacking)
- Advanced feature engineering (sentiment, readability metrics)
- Domain-specific fine-tuning
- Production deployment (API endpoint)
- Real-time model updates

## Files

```
fake-news-classification-nlp/
├── notebooks/
│   └── fake_news_classification.ipynb    # Main analysis notebook
├── data/
│   └── fake_news_data.xlsx               # Dataset
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
└── .gitignore                             # Git configuration
```

## License

MIT License

## Author

Nipun

# Fake News Classification using Classical NLP

A binary text classification system that distinguishes fake news from factual reporting using classical NLP techniques and machine learning.

## Overview

This project implements fake news detection using:
- **Preprocessing**: Tokenization, stemming, and stopword removal
- **Feature Engineering**: TF-IDF vectorization and linguistic analysis
- **Models**: Logistic Regression and Linear SVM
- **Evaluation**: Standard classification metrics and comparison

## Dataset

The dataset contains news articles with the following columns:
- `title`: Article headline
- `text`: Full article content
- `date`: Publication date
- `fake_or_factual`: Label (Fake News or Factual News)

**Data Split**: 70% training, 30% test
**Test Set Size**: 60 articles (27 Factual News, 33 Fake News)

## Methodology

### 1. Data Preprocessing
- Convert text to lowercase
- Remove special characters and digits
- Tokenize into words
- Remove stopwords (English)
- Apply Porter stemming

### 2. Feature Extraction
- POS (Part-of-Speech) tagging using spaCy
- Named Entity Recognition (NER)
- Linguistic feature analysis
- TF-IDF vectorization for model training

### 3. Model Training
Two classical classifiers are trained and compared:

| Model | Type | Algorithm |
|-------|------|-----------|
| Logistic Regression | Linear Classifier | Probabilistic learning |
| Linear SVM | Support Vector Machine | Hinge loss (SGDClassifier) |

### 4. Evaluation
Models are evaluated on the test set using standard metrics.

## Results

### Model Performance on Test Set

**Linear SVM:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Factual News | 0.79 | 0.96 | 0.87 | 27 |
| Fake News | 0.96 | 0.79 | 0.87 | 33 |
| **Accuracy** | | | **0.87** | 60 |
| Macro Avg | 0.88 | 0.88 | 0.87 | 60 |
| Weighted Avg | 0.88 | 0.87 | 0.87 | 60 |

**Logistic Regression:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Factual News | 0.74 | 0.96 | 0.84 | 27 |
| Fake News | 0.96 | 0.73 | 0.83 | 33 |
| **Accuracy** | | | **0.83** | 60 |
| Macro Avg | 0.85 | 0.85 | 0.83 | 60 |
| Weighted Avg | 0.86 | 0.83 | 0.83 | 60 |

### Key Observations

1. **Linear SVM outperforms Logistic Regression** with 87% accuracy vs 83%
2. **Precision-Recall Trade-off**:
   - SVM achieves better overall balance
   - LR has higher recall for factual news (0.96 vs 0.96) but lower precision
3. **Common Nouns in Fake News**: People, President, Women, Time, Campaign, Government, Law, Year, State, Election
4. **Common Nouns in Factual News**: Government, Year, State, Bill, Administration, President, Election, People, Order, Law

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fake-news-classification-nlp.git
cd fake-news-classification-nlp

# Install dependencies
pip install -r requirements.txt

# Download NLTK and spaCy data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Run the Jupyter notebook
jupyter notebook notebooks/fake_news_classification.ipynb
```

The notebook includes:
- Data loading and exploration
- Text preprocessing with tokenization and stemming
- POS tagging and named entity extraction
- Linguistic feature analysis
- TF-IDF feature extraction
- Model training (Logistic Regression + Linear SVM)
- Performance evaluation with confusion matrices
- Classification reports

## Tech Stack

- **Data Processing**: Pandas, NumPy
- **NLP**: NLTK, spaCy
- **Feature Extraction**: Scikit-learn (TfidfVectorizer)
- **Modeling**: Scikit-learn (LogisticRegression, SGDClassifier)
- **Visualization**: Matplotlib, Seaborn

## Limitations & Future Work

### Current Limitations
- Limited to English text only
- Binary classification (Fake vs Factual)
- Fixed preprocessing pipeline
- No model persistence/serialization
- No real-time prediction capability

### Future Improvements
- Hyperparameter tuning using GridSearchCV
- Ensemble methods (voting classifier, stacking)
- Advanced feature engineering (sentiment, readability, linguistic complexity)
- N-gram analysis and Topic modeling (LDA)
- Cross-validation for robust evaluation
- Model explainability (feature importance, SHAP values)
- Production-ready API deployment

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

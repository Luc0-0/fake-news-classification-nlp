# <u>Fake News Classification using Classical NLP</u>

This project explores fake news detection using **classical NLP techniques** and **linear machine learning models**.  
The goal is to build a **clear and interpretable pipeline**, focusing on fundamentals rather than complex architectures.

---

## <u>Overview</u>

The task is a **binary text classification problem**:
- **Fake News**
- **Factual News**

The project follows a traditional NLP workflow:
- text preprocessing
- feature extraction (Bag-of-Words / TF-IDF)
- training linear classifiers
- evaluating performance using standard metrics

---

## <u>Dataset</u>

The dataset consists of news articles with the following fields:

- `title` – article headline  
- `text` – full article content  
- `date` – publication date  
- `fake_or_factual` – label (Fake News / Factual News)

**Data split**
- Training: 70%
- Testing: 30%

**Test set size**
- 60 articles  
  - 27 Factual News  
  - 33 Fake News  

---

## <u>Methodology</u>

### 1. <u>Text Preprocessing</u>
The text is cleaned using a simple and transparent pipeline:
- lowercasing
- removal of special characters and digits
- tokenization
- stopword removal (English)
- Porter stemming

This keeps the feature space compact and interpretable.

---

### 2. <u>Feature Extraction</u>
The following linguistic features are explored:
- Bag-of-Words / TF-IDF representations
- Part-of-Speech (POS) tagging
- Named Entity Recognition (NER)
- basic linguistic frequency analysis

TF-IDF features are used for model training.

---

### 3. <u>Models</u>
Two linear classifiers are trained and compared:

- **Logistic Regression**
- **Linear Support Vector Machine (SGDClassifier)**

These models were chosen because they perform well on sparse text data and are easy to interpret.

---

### 4. <u>Evaluation</u>
Models are evaluated on the test set using:
- accuracy
- precision
- recall
- F1-score

---

## <u>Results</u>

Both models perform reasonably well on the test set:

- **Logistic Regression accuracy:** ~83%
- **Linear SVM accuracy:** ~87%

The Linear SVM shows a better balance between precision and recall, particularly for detecting fake news, and is therefore the preferred model in this setup.

---

## <u>Key Observations</u>

- Linear models are strong baselines for text classification
- Feature representation has a significant impact on performance
- Accuracy alone is insufficient; recall and F1-score provide better insight
- Fake and factual news differ in the frequency of certain nouns and named entities

---

## <u>Project Structure</u>

```
fake-news-classification-nlp/
├── notebooks/
│   └── fake_news_classification.ipynb
├── data/
│   └── fake_news_data.xlsx
├── README.md
├── requirements.txt
└── .gitignore
```

---

## <u>Tech Stack</u>

- Python  
- Pandas, NumPy  
- NLTK, spaCy  
- Scikit-learn  
- Matplotlib, Seaborn  

---

## <u>Limitations & Future Work</u>

This project is intended as a **classical NLP baseline**.

Possible next steps:
- TF-IDF hyperparameter tuning
- n-gram feature exploration
- cross-validation on a larger dataset
- comparison with more advanced models

---

## <u>License</u>

MIT License

---

## <u>Author</u>

Nipun

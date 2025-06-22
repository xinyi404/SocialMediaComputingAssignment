# Financial News Sentiment Analysis
This project explores sentiment analysis on financial news headlines using Natural Language Processing (NLP), traditional machine learning (ML), and transformer-based deep learning models. The goal is to classify sentiment (positive, neutral, negative) from a retail investor’s perspective, enabling better understanding of market perception and its implications.

## Dataset
- Source: [Kaggle – Financial News Sentiment Dataset](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- Label distribution handled using SMOTE
- Financial phrases like “interest_rate” and “quarterly_results” preserved for contextual integrity

## Techniques Used
- **Preprocessing**: Tokenization, lemmatization, financial-specific stop word filtering
- **Vectorization**: TF-IDF with n-grams (unigram, bigram, trigram)
- **Class Imbalance Handling**: SMOTE oversampling
- **Models**: Logistic Regression, SVC, Random Forest, KNN, Naïve Bayes
- **Deep Learning**: Fine-tuned BERT (via HuggingFace)

## Key Features
- Aspect-Based Sentiment Analysis (ABSA)
- Dependency-based Opinion Mining (spaCy)
- Comparative performance metrics across ML and BERT models
- Rich data visualizations and confusion matrices

## Requirements
- Python 3.8+
- `scikit-learn`, `nltk`, `pandas`, `matplotlib`, `seaborn`, `transformers`, `spaCy`, `TextBlob`

---

## Future Work
- Further fine-tune BERT with financial-domain-specific pretraining
- Use ensemble models combining SVC and BERT for robustness
- Incorporate multimodal data (e.g., stock trends, tweet sentiments)
- Deploy a real-time sentiment analysis dashboard for analysts


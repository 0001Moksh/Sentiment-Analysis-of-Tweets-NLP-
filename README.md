# 🚀 Advanced Sentiment Analysis Project

## Unveiling Emotions from Text using Deep Learning & Machine Learning

-----

## ✨ Project Overview

This project delves into the fascinating world of **Sentiment Analysis**, aiming to automatically classify text as either positive or negative. Leveraging a massive dataset of 1.6 million tweets, we explore and compare the effectiveness of two distinct machine learning paradigms:

1.  **Classical Machine Learning** with Logistic Regression.
2.  **Deep Learning** with Long Short-Term Memory (LSTM) neural networks.

Our goal is to build robust models capable of understanding the underlying sentiment in written communication, which has wide applications in social media monitoring, customer feedback analysis, and more\!

-----

## 🌟 Key Features

  * **Robust Text Preprocessing**: Includes lowercasing, URL/mention removal, tokenization, stop word filtering, and lemmatization for clean, model-ready text.
  * **Dual Model Approach**: Implements and evaluates both Logistic Regression (ML) and LSTM (DL) for comparative analysis.
  * **TF-IDF Vectorization**: Utilizes TF-IDF for feature extraction in the Logistic Regression model.
  * **Word Embeddings & Sequences**: Leverages Keras Tokenizer and `pad_sequences` for efficient text representation in the LSTM model.
  * **Model Persistence**: Saves trained models (`.pkl` and `.h5` formats) and the TF-IDF vectorizer for easy deployment and future use.
  * **Performance Evaluation**: Comprehensive evaluation metrics including accuracy, classification report, and confusion matrix visualization.
  * **Scalable**: Designed to handle large datasets effectively, demonstrated with 1.6 million data points.

-----

## 🛠️ Technologies & Libraries

  * `Python 3.x`
  * `pandas` - Data manipulation and analysis.
  * `numpy` - Numerical operations.
  * `nltk` - Natural Language Toolkit for text preprocessing (stopwords, tokenization, lemmatization).
  * `re` - Regular expressions for text cleaning.
  * `scikit-learn` - Machine Learning tools (TF-IDF, Logistic Regression, train-test split, metrics).
  * `tensorflow` / `keras` - Deep Learning framework for building and training LSTM model.
  * `matplotlib` - Data visualization (e.g., confusion matrix).
  * `seaborn` - Enhanced data visualization.
  * `joblib` - Model persistence (saving/loading models and vectorizers).

-----

## 🚀 Getting Started

Follow these steps to set up the project locally and run the sentiment analysis:

### Prerequisites

  * Python 3.8+
  * Pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/sentiment-analysis-project.git
    cd sentiment-analysis-project
    ```

    (Replace `your-username` with your actual GitHub username)

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install pandas numpy scikit-learn nltk tensorflow matplotlib seaborn joblib
    ```

4.  **Download NLTK data:**
    You'll need to download specific NLTK corpora. This is handled within the provided Colab notebook, but if running locally, ensure these are available:

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt') # The code uses 'punkt_tab' but 'punkt' is generally sufficient for tokenization.
    nltk.download('wordnet')
    ```

### Dataset

The project uses the "Sentiment140" dataset.

  * **Download Link**: [Kaggle: Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/datasets/kazanova/sentiment140)
  * **File Name**: `training.1600000.processed.noemoticon.csv`
  * **Placement**: Place this CSV file in a directory accessible by your script. In the provided Colab notebook, it's expected at `/content/drive/MyDrive/Colab Notebooks/training.1600000.processed.noemoticon.csv`. Adjust the path in your code if running locally.

### Running the Project

The core logic is demonstrated in a Jupyter Notebook (or Google Colab notebook).

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  Open the notebook file (e.g., `sentiment_analysis.ipynb` if you save the code into one).
3.  Run all cells sequentially. The notebook will:
      * Load and preprocess data.
      * Train the Logistic Regression model.
      * Evaluate Logistic Regression.
      * Train the LSTM model.
      * Evaluate LSTM.
      * Save both models and the TF-IDF vectorizer.

-----

## 📊 Results & Performance

This project demonstrates the training and evaluation of two distinct models for sentiment analysis:

### Logistic Regression Model

  * **Accuracy:** `~77.8%`
  * **Key Insight:** A strong baseline model for text classification, showing good performance with TF-IDF features.

<!-- end list -->

```
              precision    recall  f1-score   support

           0       0.79      0.76      0.77    159494
           1       0.77      0.80      0.78    160506

    accuracy                           0.78    320000
   macro avg       0.78      0.78      0.78    320000
weighted avg       0.78      0.78      0.78    320000
```

#### Confusion Matrix (Logistic Regression)

*(Note: You'll need to generate and save this image in your repo, or add a placeholder if you don't save it directly.)*

### LSTM Deep Learning Model

  * **Accuracy:** `~79.8%`
  * **Key Insight:** The LSTM model shows a slight but notable improvement over Logistic Regression, indicating the power of deep learning for capturing complex patterns in sequential text data.

<!-- end list -->

```
LSTM Accuracy: 0.798453152179718
```

-----

## 📂 Project Structure

```
.
├── sentiment_analysis_notebook.ipynb  # Main Google Colab/Jupyter Notebook with all code
├── sentiment_lr_model.pkl             # Saved Logistic Regression model
├── tfidf_vectorizer.pkl               # Saved TF-IDF Vectorizer
├── sentiment_lstm_model.h5            # Saved LSTM Deep Learning model
├── training.1600000.processed.noemoticon.csv # Dataset (needs to be downloaded separately)
├── README.md                          # This file
└── images/                            # Optional: Directory for plots/screenshots
    └── confusion_matrix_lr.png        # Placeholder for generated confusion matrix
```

*(Note: The `.pkl` and `.h5` files will be generated after running the notebook. You might want to add them to `.gitignore` if they are very large and you only want to track code.)*

-----

## 🤝 Contributing

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

-----

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

## 🧑‍💻 Author

**Deva** (AI Assistant created by Moksh Bhardwaj)

  * **Moksh Bhardwaj** - B.Tech AIML Student | DPG ITM College
  * **Location**: Basai Enclave Part 3, Gurugram, Haryana, India

-----

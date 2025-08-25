# Sentiment Analysis on IMDB Reviews

Welcome to the **Sentiment Analysis on IMDB Reviews** project! This repository contains the implementation and analysis of three text representation techniques—**TF-IDF**, **Word2Vec**, and **BERT**—for performing binary sentiment classification (positive or negative) on the IMDB movie review dataset. The goal of this project is to evaluate the effectiveness of these methods in terms of accuracy, precision, recall, and F1-score, while exploring their strengths and limitations in a practical setting.

## Project Overview
Sentiment analysis is a fundamental task in Natural Language Processing (NLP) that involves classifying text based on its emotional tone. In this project, we use the IMDB dataset, which consists of 50,000 movie reviews labeled as positive or negative. We implemented three distinct text representation approaches:

- **TF-IDF**: A traditional method that captures word importance based on term frequency and inverse document frequency.
- **Word2Vec**: A word embedding technique that encodes semantic relationships between words.
- **BERT**: A state-of-the-art transformer model that captures deep contextual information.

Each method was paired with appropriate classifiers (e.g., Logistic Regression for TF-IDF, neural networks for Word2Vec and BERT) and evaluated on standard performance metrics. The results provide insights into the trade-offs between simplicity, semantic richness, and computational complexity.

## Key Findings
- **TF-IDF** achieved the highest accuracy (**87.9%**) and excelled in precision, recall, and F1-score, proving to be a robust and efficient baseline for sentiment analysis.
- **Word2Vec** performed well (**83.1% accuracy**) by capturing semantic relationships, though it was slightly less effective due to the loss of contextual nuance in document-level embeddings.
- **BERT** underperformed (**75.2% accuracy**) due to inadequate fine-tuning and resource constraints, highlighting its dependency on extensive computational resources.
- **Challenges**: All models struggled with reviews containing mixed sentiments or sarcasm, underscoring the limitations of current NLP techniques in capturing nuanced tones.

For a detailed analysis, see the [Conclusion](Conclusion.md) file.

## Repository Structure
```
├── data/
│   ├── imdb_dataset.csv        # IMDB dataset (not included; download from Kaggle)
├── notebooks/
│   ├── tfidf_analysis.ipynb    # TF-IDF implementation and evaluation
│   ├── word2vec_analysis.ipynb # Word2Vec implementation and evaluation
│   ├── bert_analysis.ipynb     # BERT implementation and evaluation
├── src/
│   ├── preprocess.py           # Data preprocessing scripts
│   ├── models.py               # Model definitions and training logic
├── Conclusion.md               # Detailed analysis and insights
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or Google Colab
- Required libraries (listed in `requirements.txt`)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25n/imdb-dataset-of-50k-movie-reviews) and place it in the `data/` directory.

### Running the Code
1. Open the Jupyter notebooks in the `notebooks/` directory to explore the implementations:
   - `tfidf_analysis.ipynb`: TF-IDF with Logistic Regression
   - `word2vec_analysis.ipynb`: Word2Vec with a neural network
   - `bert_analysis.ipynb`: BERT with fine-tuning
2. Follow the instructions in each notebook to preprocess the data, train the models, and evaluate their performance.

## Results
| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| TF-IDF      | 87.9%    | 0.875     | 0.885  | 0.880    |
| Word2Vec    | 83.1%    | 0.826     | 0.839  | 0.833    |
| BERT        | 75.2%    | 0.000     | 0.000  | 0.000    |

For a deeper discussion of these results, refer to the [Conclusion](Conclusion.md) file.

## Usage Notes
- **TF-IDF** is recommended for quick, high-performing sentiment analysis with limited computational resources.
- **Word2Vec** is suitable for tasks requiring semantic understanding but may struggle with longer texts.
- **BERT** requires significant computational resources and fine-tuning expertise. Ensure access to a GPU and sufficient training time for optimal results.
- The preprocessing scripts in `src/preprocess.py` handle text cleaning, tokenization, and data preparation.

## Future Improvements
- Fine-tune BERT with optimized hyperparameters and more training epochs.
- Explore hybrid approaches combining TF-IDF and embeddings for improved performance.
- Address misclassifications in reviews with sarcasm or mixed sentiments using advanced contextual models.
- Experiment with other transformer-based models like RoBERTa or DistilBERT.

## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests to improve the code, documentation, or analysis.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The IMDB dataset is provided by [Kaggle](https://www.kaggle.com/datasets/lakshmi25n/imdb-dataset-of-50k-movie-reviews).
- Thanks to the open-source communities behind scikit-learn, Gensim, and Hugging Face for their excellent libraries.
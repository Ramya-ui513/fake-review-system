# Fake Review Detection System

## Abstract

This research project implements a machine learning-based system for detecting fake product reviews. The system uses natural language processing techniques and ensemble learning algorithms to classify reviews as genuine or fake based on textual features, sentiment analysis, and statistical patterns.

## Research Objectives

- Develop an automated system to identify fraudulent product reviews
- Analyze linguistic and statistical features that distinguish fake reviews from genuine ones
- Implement and compare multiple machine learning models for classification
- Create a web-based interface for real-time review analysis

## Dataset

The project uses a comprehensive dataset of product reviews labeled as genuine (CG) or fake (OR). The dataset includes reviews from various product categories with features such as review text, ratings, product categories, and metadata.

Dataset file: `fake reviews dataset.csv`

## Methodology

### Feature Engineering

The system extracts multiple feature types from each review:

1. **Textual Features**
   - Text length
   - Word count
   - Exclamation and question mark frequency
   - Capital letter ratio
   - Punctuation count

2. **Sentiment Analysis**
   - Positive sentiment score
   - Negative sentiment score
   - Neutral sentiment score
   - Compound sentiment score

3. **TF-IDF Vectorization**
   - Term frequency-inverse document frequency representation of review text
   - Captures important keywords and phrases

4. **Categorical Features**
   - Product category encoding
   - Rating information

### Machine Learning Models

The research implements and evaluates the following algorithms:

- CatBoost Classifier
- XGBoost Classifier
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)

The trained model is serialized and saved as `fake_review_detection_model.pkl`.

## Project Structure

```
├── app.py                          # Main Flask application
├── fake_review_app.py              # Alternative Flask application version
├── Fake_Review_Detection.ipynb    # Jupyter notebook with model training
├── fake reviews dataset.csv        # Training dataset
├── fake_review_detection_model.pkl # Trained model file
├── requirements.txt                # Python dependencies
├── run_app.bat                     # Windows batch script to launch app
├── sample_batch_test.csv           # Sample data for batch testing
├── sample_reviews.txt              # Sample individual reviews
├── datasetlink.txt                 # Dataset source information
├── templates/
│   └── index.html                  # Web interface template
└── catboost_info/                  # CatBoost training logs
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone or download the project repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Verify NLTK data downloads (handled automatically on first run):
   - punkt tokenizer
   - stopwords corpus
   - vader lexicon for sentiment analysis

## Usage

### Running the Web Application

**Method 1: Using Python**
```bash
python app.py
```

**Method 2: Using Batch File (Windows)**
```bash
run_app.bat
```

The application will start on `http://localhost:5000`

### Web Interface Features

1. **Single Review Analysis**
   - Enter review text
   - Optional: Specify rating and product category
   - Get instant classification result with confidence score

2. **Batch Processing**
   - Upload CSV file with multiple reviews
   - Process all reviews simultaneously
   - Download results with predictions

### API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Single review prediction
- `POST /batch-predict` - Batch review processing

## Model Training

To retrain the model with new data:

1. Open `Fake_Review_Detection.ipynb` in Jupyter Notebook
2. Load your dataset
3. Run all cells to:
   - Preprocess data
   - Extract features
   - Train models
   - Evaluate performance
   - Save the best model

## Results and Evaluation

The model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

Detailed results are documented in the Jupyter notebook.

## Technical Implementation

### Text Preprocessing

```python
- Convert to lowercase
- Remove special characters and numbers
- Remove extra whitespace
- Tokenization (when needed)
```

### Feature Vector Structure

The final feature vector includes:
- TF-IDF features (variable dimension)
- Numerical features: 7 dimensions
- Encoded categorical features

### Model Pipeline

1. Text cleaning
2. Feature extraction
3. Model prediction
4. Confidence score calculation
5. Result presentation

## Dependencies

- Flask 3.0.0 - Web framework
- NumPy 1.24.3 - Numerical computations
- Pandas 2.0.3 - Data manipulation
- Scikit-learn 1.3.0 - Machine learning algorithms
- XGBoost 2.0.3 - Gradient boosting
- CatBoost 1.2 - Categorical boosting
- NLTK 3.8.1 - Natural language processing
- Werkzeug 3.0.1 - WSGI utilities

## Limitations

- Model performance depends on training data quality and diversity
- May not generalize well to domains significantly different from training data
- Requires periodic retraining to adapt to evolving fake review patterns
- Text preprocessing may lose some contextual information

## Future Work

- Incorporate deep learning models (LSTM, BERT) for better contextual understanding
- Add user behavior analysis features
- Implement real-time learning capabilities
- Expand to multi-language support
- Integrate with e-commerce platforms

## References

- Dataset source: Available in `datasetlink.txt`
- NLTK VADER Sentiment Analysis
- Scikit-learn documentation
- CatBoost and XGBoost documentation

## License

This project is developed for academic and research purposes.

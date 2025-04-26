import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # For balancing the data
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
import nltk
import os
import requests

# Download required NLTK data
nltk.download('stopwords')  # Stop words are common words (e.g., "the", "and") that are often removed
nltk.download('wordnet')  # WordNet is used for lemmatization

# Define the URL for the dataset and the local file path
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
local_zip_path = "smsspamcollection.zip"
local_txt_path = "SMSSpamCollection"

# Download the dataset if it doesn't exist
if not os.path.exists(local_txt_path):
    print("Downloading dataset...")
    response = requests.get(dataset_url)
    with open(local_zip_path, "wb") as f:
        f.write(response.content)
    print("Extracting dataset...")
    import zipfile
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    print("Dataset ready!")
    # Clean up the zip file after extraction
    os.remove(local_zip_path)

# Load the SMS Spam Collection dataset
# The dataset is in a tab-separated format where the first column is the class label ('spam' or 'ham'),
# and the second column contains the actual SMS message.
df = pd.read_csv(local_txt_path, sep='\t', header=None, names=["Class", "Message"])

# Convert class labels to binary values for classification:
# 'spam' -> 1 (malicious messages), 'ham' -> 0 (non-spam messages)
df['Label'] = df['Class'].apply(lambda x: 1 if x == 'spam' else 0)

# Define stop words (common words to ignore) and a lemmatizer (to normalize words)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean and preprocess messages
def clean_message(message):
    """
    Preprocess the given message:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove stop words
    4. Lemmatize words to their base forms
    """
    # Convert the message to lowercase
    message = message.lower()
    # Remove punctuation
    message = ''.join([char for char in message if char not in string.punctuation])
    # Split message into words and process each word
    words = message.split()
    # Remove stop words and lemmatize remaining words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Recombine processed words into a cleaned message
    return ' '.join(words)

# Apply the cleaning function to all messages in the dataset
df['Cleaned_Message'] = df['Message'].apply(clean_message)

# Use the Bag of Words approach with tf-idf for feature extraction:
# This step transforms text into numerical data, representing word importance.
# Both unigrams (single words) and bigrams (two consecutive words) are considered.
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Extract unigrams, bigrams, and trigrams
X = tfidf_vectorizer.fit_transform(df['Cleaned_Message']).toarray()  # Convert text to feature vectors

# Split the dataset into training and testing subsets:
# The testing set contains 2% of the data (for evaluation purposes), stratified by class labels.
X_train, X_test, y_train, y_test = train_test_split(X, df['Label'], test_size=0.02, random_state=42,
                                                    stratify=df['Label'])

# Balance the training data using SMOTE:
# SMOTE generates synthetic samples for the minority class to balance the dataset.
smote = SMOTE(random_state=42, sampling_strategy=1.0)  # Fully balance the classes
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define a RandomForestClassifier with reduced complexity:
# To reduce performance, fewer trees, lower depth, and stricter split criteria are used.
model = RandomForestClassifier(n_estimators=50,  # Use only 50 trees
                               max_depth=10,  # Limit tree depth to 10
                               max_features=10,  # Limit the number of features used for splitting
                               min_samples_split=30,  # Minimum samples required to split an internal node
                               min_samples_leaf=15,  # Minimum samples required at a leaf node
                               random_state=42,  # Set seed for reproducibility
                               n_jobs=-1)  # Enable parallel computation
# Train the Random Forest model on the balanced training data
model.fit(X_train_res, y_train_res)

# Predict the classes for the test set
y_pred = model.predict(X_test)

# Evaluate model performance:
# Calculate accuracy and the confusion matrix (TP, TN, FP, FN).
accuracy = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()  # Extract counts for confusion matrix

# Prepare performance metrics for display
performance_metrics = {
    "TP (True Positive)": tp,  # Spam messages correctly identified as spam
    "TN (True Negative)": tn,  # Ham messages correctly identified as ham
    "FP (False Positive)": fp,  # Ham messages incorrectly identified as spam
    "FN (False Negative)": fn,  # Spam messages incorrectly identified as ham
    "Accuracy": accuracy        # Overall accuracy of the model
}

# Print the performance metrics in a structured format
print("Performance Metrics:")
print(pd.DataFrame(performance_metrics, index=["Value"]).transpose())

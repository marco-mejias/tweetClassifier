## 🐦 Tweet Sentiment Classifier 

A custom-built Sentiment Analysis CLI tool in Python. 
I built this project to understand the math and logic behind Machine Learning classification.
It uses a Naïve Bayes algorithm built entirely from scratch to classify tweets as either Positive or Negative. 

# 📦 Technologies

* **Python**
* **Pandas**
* **NLTK**
* **Pickle**
* **Math**

# ⚙️ Features

Here's what you can do with the classifier through its interactive menu:

* **Train from Scratch:** Feed the model a CSV dataset. It will automatically balance the data, clean it, extract the vocabulary, and calculate prior and conditional probabilities.
* **'Brain' saving** The app automatically saves the trained data (`.pkl` file) so you can load it instantly next time without re-training.
* **Real-time Prediction:** Type your own sentences in the terminal and the app applies the exact same preprocessing to your input before predicting its sentiment.
* **Robust Evaluation:** Automatically splits data into Training and Testing sets (80/20 Split) to evaluate the model's accuracy on unseen data.

# 👩🏽‍🍳 The Process

I built this project iteratively, increasing the complexity to understand how data quality and mathematical formulas affect the model's performance:

1. **The Raw Engine:** I started by implementing a basic Bag of Words (BoW) model. To prevent arithmetic underflow (multiplying many tiny probabilities resulting in zero), I transformed the equations to work in log-space.
2. **Experimentation & The Noise Problem:** I tested the model with different vocabulary sizes. I quickly realized that without smoothing, the model was highly susceptible to "noise" (typos, extremely rare words). If a word in the test set had never appeared in the training set for a specific class, its probability became zero, instantly ruining the prediction.
3. **Solution with Laplace Smoothing:** To fix the previous issue, I implemented Laplace Smoothing. This mathematically ensures no word has a zero probability, making the model robust enough to handle the entire 600k+ word vocabulary without overfitting.
4. **Refactoring:** Finally, I refactored the entire code into an Object-Oriented structure, separated the training logic from the user interface (`main_def.py`), and added an interactive menu.

# 📚 What I Learned

* **🧠 ML Manual Computation:** I learned how Naïve Bayes actually calculates probabilities based on word frequencies.
* **🔢 Log-Space Math:** I learned the importance of logarithms to turn probability multiplications into safe additions so that small decimals don't crash the process.
* **🩹 Laplace Smoothing:** I learned how applying the smoothing can save the model from overfitting to noisy data.
* **🧹 Data Preprocessing:** Implementing the `preprocess_text` pipeline taught me why stemming, removing stopwords, and balancing datasets are crucial steps.

# ⚠️ Limitations & Edge Cases (The "Bag of Words" effect)

Because this model uses a basic *Bag of Words* approach, it evaluates words independently without understanding semantics, sarcasm, or context. 

For example, highly polarized words like "love" can overpower negative context:
* `"I love this beautiful day"` -> POSITIVE 🟩
* `"I love murder"` -> POSITIVE 🟩 *(The immense positive weight of the word "love" in the training data outscores the negative weight of the word "murder").*

This perfectly illustrates why modern AI uses advanced architectures (like Transformers or Word Embeddings) for deep semantic understanding, though Naïve Bayes remains incredibly fast, lightweight, and effective for general sentiment classification.

### 💭 How can it be improved?

* Implement **TF-IDF** (Term Frequency-Inverse Document Frequency) instead of raw word counts to give less weight to common words.
* Add support for **N-grams** so the model can understand phrases like "not good" instead of evaluating "not" and "good" separately.

### 🚦 Running the Project

To run the project in your local environment, follow these steps:

1. Clone the repository to your local machine.
2. Make sure you have Python installed.
3. Run `pip install pandas nltk` in the terminal to install the required dependencies.
4. *(Optional)* Place the `FinalStemmedSentimentAnalysisDataset.csv` file in the root directory if you want to train a new model from scratch.
5. Run `python main_def.py` to launch the interactive CLI.

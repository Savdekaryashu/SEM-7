# Email Spam Filtering using Naive Bayes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# Step 1: Load dataset
# -----------------------

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels: spam = 1, ham = 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# -----------------------
# Step 2: Split data
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'],
                                                    test_size=0.2, random_state=42)

# -----------------------
# Step 3: Convert text to vectors
# -----------------------
cv = CountVectorizer(stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# -----------------------
# Step 4: Train model
# -----------------------
model = MultinomialNB()
model.fit(X_train_cv, y_train)

# -----------------------
# Step 5: Predict
# -----------------------
y_pred = model.predict(X_test_cv)

# -----------------------
# Step 6: Evaluate
# -----------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------
# Step 7: Test custom email
# -----------------------
sample = ["Congratulations! You've won a $500 Walmart gift card. Click here to claim."]
sample_vec = cv.transform(sample)
print("\nPrediction for sample:", model.predict(sample_vec))  # 1 means spam

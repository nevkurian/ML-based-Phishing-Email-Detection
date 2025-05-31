import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
import time

start = time.time()

# Load dataset
print("📥 Loading dataset...")
df = pd.read_csv('Phishing_Email.csv')  # Replace with your actual CSV filename

# Clean dataset
print("🧹 Cleaning dataset...")
df = df[['Email Text', 'Email Type']].dropna()
print(f"✅ Dataset ready: {len(df)} samples")

# Convert labels to binary
print("🔁 Converting labels...")
df['Email Type'] = df['Email Type'].apply(lambda x: 0 if x.strip().lower() == 'safe email' else 1)

# Split data
print("🧪 Splitting data into train and test sets (80:20)...")
X_train, X_test, y_train, y_test = train_test_split(df['Email Text'], df['Email Type'], test_size=0.2, random_state=42)
print(f"🔢 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Vectorization
print("🔤 Vectorizing text data with TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("🤖 Training SVM model...")
model = SVC(kernel='linear', verbose=True)
model.fit(X_train_vec, y_train)
print("✅ Model training completed.")

# Evaluation
print("📊 Evaluating model...")
y_pred = model.predict(X_test_vec)
print("\n📝 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
print("💾 Saving model and vectorizer...")
with open('svm_text_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"✅ Model and vectorizer saved successfully.")
print(f"⏱️ Total training time: {time.time() - start:.2f} seconds")

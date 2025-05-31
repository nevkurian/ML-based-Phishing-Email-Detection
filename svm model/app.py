import pickle
import os
import sys
from flask import Flask, request, render_template, url_for

# Crucially, import the classes for the objects you are loading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# --- Configuration ---
# Adjust these paths if your pkl files are located elsewhere relative to app.py
MODEL_PATH = os.path.join('ML', 'svm_text_model.pkl')
VECTORIZER_PATH = os.path.join('ML', 'vectorizer.pkl')
# --- End Configuration ---

# Initialize Flask App
app = Flask(__name__)

# --- Load Model and Vectorizer ---
# Load these once when the app starts
vectorizer = None
model = None

print(f"🔄 Attempting to load vectorizer from '{VECTORIZER_PATH}'...")
if not os.path.exists(VECTORIZER_PATH):
    print(f"❌ FATAL ERROR: Vectorizer file not found at '{VECTORIZER_PATH}'")
    sys.exit(1) # Stop the app if essential file is missing
try:
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load vectorizer: {e}")
    sys.exit(1)

print(f"🔄 Attempting to load model from '{MODEL_PATH}'...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ FATAL ERROR: Model file not found at '{MODEL_PATH}'")
    sys.exit(1)
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load model: {e}")
    sys.exit(1)

if vectorizer is None or model is None:
    print("❌ FATAL ERROR: Model or Vectorizer failed to load. Exiting.")
    sys.exit(1)
# --- End Loading ---


def predict_phishing(text):
    """Predicts if the text is phishing using the loaded components."""
    if not text or not isinstance(text, str):
        return "Invalid Input"

    try:
        # Vectorizer expects an iterable (like a list)
        text_vector = vectorizer.transform([text])
        # Predict using the loaded model
        prediction = model.predict(text_vector)

        # Convert numerical prediction back to label (0=Safe, 1=Phishing)
        if prediction[0] == 0:
            return "Safe Email"
        elif prediction[0] == 1:
            return "Phishing Email"
        else:
            return "Unknown Prediction" # Should not happen
    except Exception as e:
        print(f"⚠️ Error during prediction: {e}")
        return "Prediction Error"


# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the initial HTML form."""
    # Renders the templates/index.html file
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles form submission, predicts, and shows results."""
    if request.method == 'POST':
        # Get data from the form fields using their 'name' attributes
        sender = request.form.get('sender', 'Not Provided') # Use .get for safety
        subject = request.form.get('subject', 'Not Provided')
        email_body = request.form.get('emailBody', '')

        if not email_body:
            # Handle case where body is empty, maybe return an error message
            # or redirect back with a message. For now, render result with warning.
            return render_template('result.html', prediction="No Email Body Provided", email_body=email_body)

        # Perform prediction ONLY on the email body
        prediction_result = predict_phishing(email_body)

        # Render the result page, passing the prediction and the original body
        return render_template('result.html', prediction=prediction_result, email_body=email_body)

    # If not POST (though action specifies POST), redirect to home
    return redirect(url_for('home'))

# --- Run the App ---
if __name__ == '__main__':
    # host='0.0.0.0' makes it accessible on your network
    # debug=True provides auto-reloading and detailed error pages (DO NOT use in production)
    print("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
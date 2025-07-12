
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline

st.set_page_config(layout="wide", page_title="📰 Fake News Detector with Explainability")

st.title("📰 Multi-Model Fake News Detection with Explainability (LIME)")
st.markdown("Upload a news dataset or use the default one to classify articles as **FAKE** or **REAL**, with explanations using LIME.")

# Load default dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/fake_or_real_news.csv")
    return df

df = load_data()

# Dataset view
with st.expander("📊 Dataset Preview"):
    st.write(df.head())

# Model training section
st.subheader("🧠 Train Multiple Models")

# Preprocessing
X = df['text']
y = df['label']
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train Models
models = {
    "Passive Aggressive": PassiveAggressiveClassifier(max_iter=50),
    "Logistic Regression": LogisticRegression(),
    "Linear SVM": LinearSVC()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    results[name] = report['weighted avg']['f1-score']

# Display Results
st.subheader("📈 Model F1-Scores")
st.bar_chart(results)

# Choose model for LIME explainability
st.subheader("🔍 Explain a Prediction with LIME")
sample_text = st.text_area("Enter News Text:", height=200, value=X_test[10])

chosen_model_name = st.selectbox("Select a model for explanation", list(models.keys()))
model = models[chosen_model_name]
pipeline = make_pipeline(vectorizer, model)

if st.button("Explain Prediction"):
    explainer = lime.lime_text.LimeTextExplainer(class_names=["FAKE", "REAL"])
    exp = explainer.explain_instance(sample_text, pipeline.predict_proba, num_features=10)
    st.markdown("### 🔍 LIME Explanation")
    st.components.v1.html(exp.as_html(), height=700, scrolling=True)

# Model prediction
if st.button("Predict"):
    pred = model.predict(vectorizer.transform([sample_text]))[0]
    st.success(f"The news article is **{pred.upper()}**")

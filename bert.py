import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Streamlit app title and header
st.title("BERT Text Classifier App")
st.write("This app uses BERT to classify text as either positive or negative sentiment.")

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model():
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return classifier

classifier = load_model()

# Text input from the user
user_input = st.text_area("Enter some text:")

# Classify button
if st.button("Classify Text"):
    if user_input:
        # Run the classification
        result = classifier(user_input)
        # Display the result
        st.write("Classification result:", result)
    else:
        st.write("Please enter some text to classify.")

# Run Streamlit app by using command: streamlit run your_app.py

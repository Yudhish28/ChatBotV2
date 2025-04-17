# main.py

import random
import json
import pickle
import numpy as np
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from streamlit_chat import message

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and data files
model = load_model('chatbotmodel.h5')
intents = json.load(open('breastCancer.json'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Fallback responses
fallback_responses = [
    "I'm not sure I got that. Can you try rephrasing?",
    "Sorry, I can only help with breast cancer questions.",
    "Hmm, I didn’t catch that. Ask me anything about breast cancer!",
    "I'm here to help with breast cancer-related queries. Can you be more specific?"
]

# Clean and tokenize sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert input to bag-of-words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return bag

# Predict response
def get_response(user_input):
    input_bow = bow(user_input, words)
    input_bow = np.array([input_bow])

    results = model.predict(input_bow)[0]
    max_prob = max(results)
    tag_index = np.argmax(results)

    if max_prob > 0.7:
        tag = classes[tag_index]
        for intent in intents['intents']:
            if intent['tags'] == tag:
                return random.choice(intent['responses'])
    else:
        return random.choice(fallback_responses)

# --- Streamlit UI ---
st.title("🩺 Breast Cancer Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="input")

if user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

# Display chat history
for i, (sender, msg) in enumerate(st.session_state.chat_history):
    is_user = sender == "user"
    message(msg, is_user=is_user, key=str(i))

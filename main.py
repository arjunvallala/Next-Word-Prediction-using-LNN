import tensorflow
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

max_seq_len = 14

@st.cache_resource
def load_tokenizer():
    with open("tokenizer","rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model_():
    return load_model("lstm.keras")



def predict_next_word(model, tokenizer, text, max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if not token_list:
        return None

    token_list = token_list[-(max_seq_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_seq_len - 1,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)

    predicted_word_index = int(np.argmax(predicted, axis=1)[0])

    # Convert the predicted index back to a word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None

tokenizer = load_tokenizer()
model = load_model_()

st.title("Prediction next word using LSTM")
text = st.text_input("Enter the input text here")

if st.button("Submit"):
    if text.strip():
        word = predict_next_word(model,tokenizer,text,max_seq_len)
        if word is None:
            st.header(f"No available words in training dataset")
        else:
            st.success(f"The predicted next word is: {word}")
    else:
        st.warning("Please type your input")
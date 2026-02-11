import streamlit as st
import torch
import torch.nn as nn
import pickle


st.title("Next Word Prediction")


device = "cpu"


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size + 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


@st.cache_resource
def load_model():
    with open("../model/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)

    idx2word = {i: w for w, i in word2idx.items()}

    model = LanguageModel(len(word2idx), 64, 128)
    model.load_state_dict(torch.load("../model/language_model.pth", map_location=device))
    model.eval()

    return model, word2idx, idx2word


model, word2idx, idx2word = load_model()


def predict_next(text_input, max_len=10):
    words = text_input.lower().split()
    encoded = [word2idx.get(w, 0) for w in words]

    if len(encoded) > max_len:
        encoded = encoded[-max_len:]
    else:
        encoded = [0] * (max_len - len(encoded)) + encoded

    x = torch.tensor([encoded])

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    return idx2word.get(pred, "<UNK>")


text = st.text_input("Enter text")

if st.button("Predict"):
    if text:
        st.success(f"Next word: {predict_next(text)}")

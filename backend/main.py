import torch
import torch.nn as nn
import pickle
from fastapi import FastAPI

app = FastAPI(title="Next Word Prediction API")


device = torch.device("cpu")


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


with open("../model/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

idx2word = {i: w for w, i in word2idx.items()}


model = LanguageModel(len(word2idx), 64, 128)
model.load_state_dict(torch.load("../model/language_model.pth", map_location=device))
model.eval()


def predict_next(text_input, max_len=10):
    words = text_input.lower().split()
    encoded = [word2idx.get(w, 0) for w in words]

    if len(encoded) > max_len:
        encoded = encoded[-max_len:]
    else:
        encoded = [0] * (max_len - len(encoded)) + encoded

    x = torch.tensor([encoded]).to(device)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    return idx2word.get(pred, "<UNK>")


@app.get("/")
def home():
    return {"message": "Language Model API is running"}


@app.post("/predict")
def predict(text: str):
    next_word = predict_next(text)
    return {"input": text, "next_word": next_word}

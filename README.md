# Next Word Prediction NLP App

End-to-end NLP project using PyTorch, FastAPI, and Streamlit.

## Features
- Next word prediction using LSTM
- FastAPI backend
- Streamlit frontend
- Real-time inference

## Run locally

### Install
pip install -r requirements.txt

### Start backend
cd backend
uvicorn main:app --reload

### Start frontend
cd frontend
streamlit run app.py

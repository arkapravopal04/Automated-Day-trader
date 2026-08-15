from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from engine import Tensor
from Neural_Nets import Linear
from module import Module
import requests


class NLPEncoder(Module):
    def __init__(self, hidden_size=64):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = hidden_size
        self.linear = Linear(3, hidden_size)
    def forward(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(0).numpy().astype(np.float64)
        return self.linear(Tensor(logits))
    def parameters(self):
        return self.linear.parameters()
def fetch_news(ticker, api_key, num_articles=5):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&"
        f"language=en&"
        f"sortBy=publishedAt&"
        f"pageSize={num_articles}&"
        f"apiKey={api_key}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[fetch_news] Request failed: {e}")
        return []
    if data.get('status') != 'ok':
        print(f"[fetch_news] API error: {data.get('message', 'unknown')}")
        return []
    headlines = [article['title'] for article in data.get('articles', []) if article.get('title')]
    return headlines

def get_sentiment_vector(ticker, api_key, encoder, num_articles=5):
    headlines = fetch_news(ticker, api_key, num_articles)

    if not headlines:
        return Tensor(np.zeros(encoder.hidden_size))

    vectors = [encoder(h) for h in headlines]
    stacked = np.stack([v.data.flatten() for v in vectors], axis=0)
    avg = np.mean(stacked, axis=0)
    return Tensor(avg)
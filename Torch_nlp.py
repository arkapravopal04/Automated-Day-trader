

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import numpy as np
import requests
from Torch_Neural_Nets import DEVICE


class NLPEncoder(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = hidden_size
        self.linear      = nn.Linear(3, hidden_size)

    def forward(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(0).float()  
        return self.linear(logits)            

    def parameters(self, recurse=True):
        return self.linear.parameters(recurse=recurse)


def fetch_news(ticker, api_key, num_articles=5):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&language=en&sortBy=publishedAt"
        f"&pageSize={num_articles}&apiKey={api_key}"
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
    return [a['title'] for a in data.get('articles', []) if a.get('title')]


def get_sentiment_vector(ticker, api_key, encoder, num_articles=5):
    headlines = fetch_news(ticker, api_key, num_articles)
    if not headlines:
        return torch.zeros(encoder.hidden_size, device=DEVICE)
    vectors = [encoder(h) for h in headlines]
    stacked = torch.stack(vectors, dim=0)     
    return stacked.mean(dim=0)   
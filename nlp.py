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
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(0).numpy()  
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
    response = requests.get(url)
    data = response.json()
    
    if data['status'] != 'ok':
        print(f"Error: {data['message']}")
        return []
    
    headlines = [article['title'] for article in data['articles']]
    return headlines

def get_sentiment_vector(ticker, api_key, encoder, num_articles=5):
    headlines = fetch_news(ticker, api_key, num_articles)
    
    if not headlines:
        return Tensor(np.zeros(encoder.hidden_size))
    
    vectors = [encoder(headline) for headline in headlines]
    stacked = np.stack([v.data for v in vectors])
    avg = np.mean(stacked, axis=0)
    return Tensor(avg)


# Automated Day Trader

> Building a fully autonomous trading system from scratch — no high-level ML libraries.  
> Core: custom autograd engine over a DAG with topological-sort-driven backpropagation.  
> Status: actively ongoing — NLP front-end and LSTM in progress.

---

This is a personal project that I am currently working on.
As the name suggests I aim to make a completely automated day trader i.e. a "bot" that can make decisions by itself — using some kind of NLP or LLM to understand market data and make decisions from that.

I want to actually understand what I am working on, hence the from-scratch implementation of everything. I will try to avoid the usage of any major libraries and do as much as I can from first principles.

## Autograd Engine

Automatic Differentiation Engine: implements a Directed Acyclic Graph (DAG) that automatically tracks mathematical operations to calculate gradients using the chain rule.

Topological Gradient Propagation: uses a topological sort to ensure gradients flow from the output back to the inputs in the mathematically correct order.

Modular Neural Network API: features a PyTorch-inspired Module class, allowing you to build complex architectures like LSTMs and CNNs by nesting layers.

Gradient Accumulation: correctly handles the multivariate chain rule using += logic, allowing a single Tensor to be reused across multiple branches of a network.

NumPy-Powered Backend: built from first principles using NumPy for efficient matrix operations, bridging the gap between raw calculus and high-level deep learning.

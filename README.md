This is a personal project that I am currently working on 
As the name suggests I aim to make a completely automated day trader i.e.

I want this "bot" to be able to make decissions by itself
using some kind of nlp or llm to understand market data and make decisions from that

I want to actually make and understand what I am working on hence the from scratch implementation of an Auto Grad engine 
I  will try to avoid the useage of any major libraries and try to do it all "FROM SCRATCH"

AUTOGRAD ENGINE:
Automatic Differentiation Engine: Implements a Directed Acyclic Graph (DAG) that automatically tracks mathematical operations to calculate gradients using the chain rule.
Topological Gradient Propagation: Uses a Topological Sort to ensure gradients flow from the output back to the inputs in the mathematically correct order.
Modular Neural Network API: Features a PyTorch-inspired Module class, allowing you to build complex architectures like LSTMs and CNNs by nesting layers.
Gradient Accumulation: Correctly handles the Multivariate Chain Rule using += logic, allowing a single Tensor to be reused across multiple branches of a network.
NumPy-Powered Backend: Built from first principles using NumPy for efficient matrix operations, bridging the gap between raw calculus and high-level deep learning.

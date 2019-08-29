---
template: default
use_math: true
---

Introduction
======

- **Artificial Intelligence**
: A thriving field with many practical applciations and active researh topics. The software is being used to assist in automating routine labor, understand sppech or images, make diagnoses in medicine and support basic scientific research

- **Deep Learning**
: A solution to allow computers to learn from experience and understand the world in terms of a hierarchy of concepts, with each concept defined through its relation to simpler concepts. If a graph is drawn showing how these concepts are built on top of each other, the graph is deep with many layers  
    - Appeals to a more general principle of learning *multiple levels of composition*

- **Knowledge Base**
: When artificial intelligence is hard-coded with knowledge about the world in formal languages such
    - The computer is able to reason automatically about statements in these formal language using logical inference rules

**Machine Learning**
: The ability for AI systems to acquire their own knowledge by extracting patterns from raw data

**Logistic Regression**
    - Was used to determine whether to recommend a cesarean delivery

**Naive Bayes**
    - Can separate legitimate emails from spam emails

**Representation**
: The characteristics of the training data matched to the data to classify

**Feature**
: Each piece of information included in the representation of the *thing* to make a prediction on

**Representation Learning**
: A machine learning device to discover not only the mapping from representation to output but also the representation itself

**Autoencoder**
: A representation learning algorithm that uses an autoencoder as well as a decoder
    - Trained to preserve as much information as possible when an input is run through the encoder and then the decoder

**Encoder**
: Converts input data into a different representation

**Decoder**
: Converts the new representation back into the original format

**Factors of Variation**
: The separate sources of influence that explain the observed data

**Multilayer Perceptron (MLP)**
: A mathematical function mapping some set of input values to output values

**Visible Layer**
: The layer of variables that we are able to observe

**Hidden Layers**
: The layer of variables where the values are not given in the data and instead the model must determine which concepts are useful for explaining the relationships in the observered data


## Who Should Read This Book?

- Book is organized into three parts
    - Part I introduces the basic mathematical tools and machine learning concepts
    - Part II describes the most established deep learning algorithms
    - Part III describes more speculative ideas that are widely believed to be important for future research in deep learning


## Historical Trends in Deep Learning

- Deep learning has becomes more useful as the amount of available training data has increased
- Deep learning models have grown in size over time as computer infrastructure (both hardware and software) for deep learning has improved
- Deep learning has solved increasingly complicated applications with increasing accuracy over time

### The Many Names and Changing Fortunes of Neural Networks

**Cybernetics**
: What deep learning was known as in the 1940s-1960s

**Connectionism**
: What deep learning was known as in the 1980s-1990s

**Artificial Neural Networks**
: One of the names that deep learning has gone by

**Adaptive Linear Element (ADALINE)**
: A linear model which could recognize two different categories of inputs by testing whether $f(x, w)$ was positive or negative. It simply returned the value of $f(x)$ itself to predict a real number and could also learn to predict from data

**Stochastic Gradient Descent**
: The training algorithm that was used to adapt the weights of the ADALINE model

**Linear Models**
: Models based on the $f(x, w)$ used by the perceptron  
    - Have many limitations
        - Cannot learn the XOR function where $f([0, 1], w) = 1$ and $f([1, 0], w) = 1$ but $f([1, 1], w) = 0$ and $f([0, 0], w) = 0$

**Rectified Linear Unit**
: What most neural networks are based on today

**Parallel Distributed Processing**
: Synonymous to connectionism

**Distributed Representation**
: The idea that each input to a system should be represented by many features and each feature should be involved in the representation of many possible inputs


### Increasing Dataset Sizes

- A general rule of thumb is that a supervised deep learning algorithm will generally achieve acceptable performance with around 5,000 labeled examples per category and will match or exceed human performance when train with a dataset containing at least 10 million labeled examples


### Increasing Model Sizes

- More neurons means better models


### Increasing Accuracy Complexity and Real-World Impact

**Reinforcement Learning**
: An autonomous agent must learn to perform a task by trial and error without any guidance from the human operator
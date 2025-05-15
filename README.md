# MachineLearning_and_DeepLearning

ML is a subset of AI. It focuses on algorithms that allow computers to learn patterns from data and improve over time without being explicitly programmed.

How it works:

Data is fed into an algorithm

The algorithm finds patterns

It makes predictions or decisions based on new input

Examples:

Spam filters in email

Product recommendations on Amazon

Stock price predictions

Methods ML:

- Supervised learning (e.g., classification) - a type of ML  where the model is trained on a labeled dataset — meaning each example in the training data includes both the input and the correct output
      Needs human input.
- Unsupervised learning (e.g., clustering) - Unsupervised Learning is a type of machine learning where the model is given input data without labeled outputs — meaning, there are no correct answers provided during training.

The goal is for the algorithm to find patterns, structures, or groupings in the data on its own.

- Reinforcement learning (e.g., game AI, robotics) - Learns by trial and error
   Rewarded when correct then penalized when it makes a mistake.

  # Difference Decision tree, logistic regression and Linear regress

A Decision Tree is a model that splits data into branches based on decision rules, ultimately arriving at a prediction (leaf). It works like a flowchart.

 Used for:
Classification (e.g., is this email spam?)

Regression (e.g., predict house price)

 How it works:
Start at a root node (feature)

Split the data based on feature values

Keep splitting until you reach a leaf node (final decision/prediction).

- Linear Regression is used to predict a numeric (continuous) value based on one or more features.

 How it works:
It fits a straight line (or hyperplane for multiple features) through the data to minimize the error between predicted and actual values.

- Logistic Regression is used for classification, especially binary classification (yes/no, 0/1, spam/not spam).

Despite the name, it's a classification algorithm — not regression!
It Uses the logistic (sigmoid) function to output a value between 0 and 1 (a probability)
Predicts whether an input belongs to a class.

Classical machine learning can be outperformed, at some tasks, by newer methods that are part of the deep learning ecosystem. But there are still reasons to use classical machine learning. These include:

- Work with structured data
- Lower expense to operate
- Easier to interpret

# How neural networks are inspired by the human brain?

An artificial neuron (also called a node or perceptron) receives one or more inputs, multiplies them by weights, adds a bias, and passes the result through an activation function.

Just like biological neurons, if the signal is strong enough (based on weighted input), the artificial neuron "fires" a signal to the next layer.

# Deep Learning
- is a subfield of machine learning that uses artificial neural networks with many layers to learn patterns from large, complex datasets.

- It is called "deep" because of the multiple layers (called hidden layers) between the input and output.

-Deep learning excels at tasks like image recognition, natural language processing, and speech recognition.

 # Key Components of the Deep Learning Ecosystem
1. Neural Network Architectures
These define how models are structured:

Feedforward Neural Networks (basic structure)

Convolutional Neural Networks (CNNs) — great for image data

Recurrent Neural Networks (RNNs) — designed for sequences (e.g., time series, text)

Transformers — state-of-the-art models for language and vision tasks

Autoencoders — used for unsupervised learning and dimensionality reduction

Generative Adversarial Networks (GANs) — for generating data (images, text, etc.)



# FromLinearFunctionToNeuralNetwork
This is a repository for Course
"From Linear Functions to Neural Networks

"01. Linear Functions: How Perceptrons Describe Intuition"

📌 How do you describe thinking and perception? --function
📌 How do you build simple cognition? --McCulloch-Pitts model (M-P model)
📌 How do you transition from intuitions that deviate from reality to perceptions that match reality through learning? --Rosenblatt perceptron model

"02. The variance cost function: knowledge of error"

📌 How are errors assessed? Direct subtraction, absolute value, squaring, etc.
📌 What is a cost function? The difference with the prediction function
📌 Variance cost function (i.e., mean square error MSE)
📌 How to find the ideal w-value

"03. Gradient descent and backpropagation: can change (up)"

📌 How to move and automatically find the lowest point of a parabola? -- Determine the direction of movement based on the slope
📌 You can tell which way to move based on the slope; how much should you move each time? --Fixed step vs. gradient descent
📌 Types of gradient descent - batch, random, mini-batch

"04. Gradient descent and backpropagation: being able to change (next)"

📌 Improve the prediction function y=wx → y=wx+b - the cost function will change from 2D to 3D
📌 How to find the fastest descending direction for a three-dimensional cost function: find partial derivatives with respect to w and partial derivatives with respect to b, introducing the concept of gradient
📌 Review to summarize what we've learned so far: predictive function models, forward propagation, cost function, backpropagation, learning rate, training

"05. Activation functions: breathing soul into the machine"

📌 Linear functions can't solve classification problems, what to do? --Introduce the activation function (Sigmoid function)

📌 How to update the parameters by gradient descent on the cost function of a predictive model that incorporates an activation function that is a sigmoid-chain derivation of composite functions

📌Summary: Mainly introduced chain derivation rule, and activation function.

"06. Hidden Layer: Why Neural Networks WORK"

📌Now the problem: Beans become not simply the bigger or smaller the more likely they are to be poisonous, but poisonous in a certain range of sizes and non-toxic in certain ranges.

📌 How can we make predictive models capable of producing more complex curves? --It's time for the neurons to form a network, add hidden layers

📌What is a hidden layer

📌 How neural networks perform gradient descent to update parameters - chaining composite functions for derivation (so tiring 😩)]

"07. Higher dimensional spaces: how machines face increasingly complex problems"

📌Summary: This lesson begins with an introduction to multidimensional features and linearly indivisible problems.
📌 Consider the colour and size of a bean, now that there are two input variables, x1 and x2. then the relationship between the toxic probability and the color and size of the bean requires a three-dimensional coordinate system.
📌 The size colour depth and toxicity distribution of the beans in the environment are linearly indistinguishable, so what to do - linearly indistinguishable problems
📌 When the problem data features more and more, it is impossible to visualize what to do with the problem - the next lesson will introduce mathematical tools for dealing with multidimensional data [vectors and matrices]
📌 [Added] If the distribution of the probability of a bean being toxic and the size colour shade is a circle. The ones inside the circle are poisonous while the ones outside the circle are not, think about how many neurons the hidden layer needs at least to twist the segmentation line into a circle to achieve type segmentation.
"08. Beginning to know Keras: easy to complete the neural network modelling"

📌 This lesson is a watershed moment, after which we will no longer analyse the underlying mathematical principles. It's time to start practising with the framework directly.
📌 An introduction to linear algebra: one dimension is a vector, two is a matrix, and three or more is a tensor. Vectors can be succinctly displayed as multivariate equations, with a single equation 
 table to represent the forward propagation of an entire layer of a neural network.
📌 Keras framework understanding and getting started
📌 Quick implementation of a few neural networks from earlier in the course using the Keras framework
📌 "09. Deep Learning: The Amazing DeepLearning"

📌 The problem with deep learning: it's hard to analyze what each node in the network is understanding with precise mathematical means
📌 Tensorflow playground play: learn how to tune the parameters, adjust the number of neurons, activation function, number of hidden layers, learning rate, etc.
📌 Problems with sigmoid activation functions: gradient vanishing
📌ReLU activation function: Dead ReLU Problem, leaky ReLU
📌 Summarize the choice of activation function: sigmoid is only suitable for the output layer for binary classification, hidden layer is suitable for ReLU
📌keras differentiate mosquito data
"10. Convolutional neural networks: breaking the bottleneck of image recognition"

📌Summary: This class is mainly about the limitations of fully connected neural network image recognition, and with the help of convolution can improve the recognition effect, but the practical part of the talk is still using fully connected neural networks to train image recognition, the data used is the mnist dataset, and the next class is the convolutional neural network practical.
📌 Difficulty of image recognition: This is no longer a problem that applies to the mechanical logic of the computer to make judgments, we need to do this with a system that has a certain degree of fault tolerance.
📌 How do fully connected neural networks perform image recognition? The training set achieves 100% correctness, while the test set is only 97.82%
📌 Fully connected neural networks are not very good at image recognition, and for this reason, convolutional neural networks are introduced: See the next section for a practical example.
📌 Emerging concepts: training and test sets, generalization, unique heat coding, softmax layer, cross-entropy, convolutional kernel.
"11. Convolutional Neural Networks: Image Recognition in Action

📌Summary: This lesson began to use convolutional neural networks for image recognition, convolutional neural networks, although it seems that there are more steps, the actual use of fewer parameters, for image recognition, has better results. Through this study, I also found that deep learning is a mystery, the number of layers, the number of neurons, the choice of various functions, etc., but also must understand the basic principles, not adjust blindly. There is also their computer want to train more epoch can not do, need to use Google Colab to alchemy, and then save the model locally.
📌 Convolutional Process Learning
📌 LeNet-5 convolutional neural network model: recognition acc98.4%
📌 Self-built convolutional neural network model: recognizing acc99.21%
"12. Loop: Sequence Dependent Problems.

📌 Summary: Analyze whether the review data of a shopping website is positive or negative

📌 Basic Learning

Words are the basic unit of natural language processing
Jieba stuttering module participle
Word vectors encode sentences
Embedding layer
Multi-categorization problems are to softmax activation function with a categorical cross-entropy function to use, while binary classification problems use sigmoid activation function with a binary cross-entropy function
📌 Word embedding matrix + fully connected layer model construction steps

"13. LSTM Networks: Natural Language Processing in Practice".

📌 Summary: The last lesson on the word vector input neural network is a direct draw flat to a first-order array, this lesson is to consider the relationship between the sequence time, introduce recurrent neural networks as well as the LSTM network, and reuse the LSTM network as well as the use of other people's word vector matrices for text sentiment analysis. 
📌 What is a recurrent neural network
📌 Long Short-Term Memory Networks LSTM (Long Short-Term Memory: ordinary recurrent neural networks don't work well for problems with long dependencies. Long Short-Term Memory (LSTM) solves the long dependency problem by introducing cell states that allow the network to remember and forget past words.
A variant of 📌LSTM: GRU
📌 On the parameters of recurrent neural networks
📌 Hands-on LSTM training for judging sentiment text
"14. Machine Learning: the last and first lesson"

📌 Review of previous lessons
📌 Relationship between artificial intelligence, machine learning, and deep learning
📌 How you should continue learning subsequently

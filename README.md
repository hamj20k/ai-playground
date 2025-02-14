# ai playground

welcome to the **ai playground**! this is a fun, interactive space where you can experiment with different neural network architectures, visualize their training progress, and tweak hyperparameters in real-time. whether you're a beginner or a seasoned ai enthusiast, this playground is designed to make machine learning approachable, intuitive, and, most importantly, *cool*. please remember, everything here should be considered a work in progress!

---

## what's this all about?

this project is a web-based interface for training and visualizing neural networks. it supports three types of models:

1. **mlp (multi-layer perceptron)**: a classic feedforward neural network for structured/tabular data.
2. **cnn (convolutional neural network)**: perfect for image-based tasks, with convolutional layers to capture spatial patterns.
3. **rnn (recurrent neural network)**: designed for sequential data like time-series or text, with memory to handle temporal dependencies.

you can tweak hyperparameters, watch the network learn in real-time, and see how changes affect performance. it's like a sandbox for ai, where you can play, experiment, and learn.

---

## why does this exist?

machine learning can feel intimidating. the math, the jargon, the endless hyperparameters—it's easy to get lost. this project is here to make ai more accessible and fun. by visualizing the training process and letting you interact with the models, it demystifies the "black box" of neural networks.

plus, it's just *cool* to watch a neural network learn. seeing the loss decrease and accuracy improve in real-time is oddly satisfying. it's like watching a robot grow smarter right before your eyes.

---

## features

- **real-time training visualization**: watch the network learn as it trains. see the loss and accuracy update live.
- **interactive hyperparameter tuning**: adjust learning rate, batch size, epochs, and more on the fly.
- **model selection**: switch between mlp, cnn, and rnn to see how they differ.
- **neural network visualization**: see the structure of your model and how data flows through it.
- **training logs**: keep track of what's happening under the hood with detailed logs.

---

## how to use

1. **select a model**: choose between mlp, cnn, or rnn. each model is suited for different types of data.
2. **tweak the hyperparameters**: adjust the learning rate, batch size, epochs, and other settings to see how they affect training.
3. **start training**: hit the "start training" button and watch the magic happen. the network will update in real-time, and you'll see the loss and accuracy improve.
4. **visualize the results**: check out the charts to see how the model is performing. the neural network visualization will show you the structure of your model.

---

## models

### mlp (multi-layer perceptron)
- **what it's for**: structured/tabular data (e.g., predicting house prices, customer churn).
- **how it works**: a fully connected network that learns patterns in your data.
- **input shape**: `[10]` (10 features per sample).

### cnn (convolutional neural network)
- **what it's for**: image data (e.g., classifying handwritten digits).
- **how it works**: uses convolutional layers to detect spatial patterns like edges and shapes.
- **input shape**: `[9, 9, 1]` (9x9 grayscale images).

### rnn (recurrent neural network)
- **what it's for**: sequential data (e.g., time-series, text).
- **how it works**: processes data step-by-step, with memory to handle sequences.
- **input shape**: `[20, 10]` (20 time steps, 10 features per step).

---

## tech stack

- **tensorflow.js**: for building and training neural networks in the browser.
- **javascript**: for interactivity and dynamic updates.
- **html/css**: for the user interface and styling.
- **chart.js**: for visualizing training progress.

---

## why this is cool

- **no setup required**: everything runs in your browser. no need to install anything.
- **real-time feedback**: see the impact of your changes immediately.
- **educational**: learn how neural networks work by experimenting with them.
- **fun**: it's just cool to play with ai, right?

---

## how to contribute

whether you want to add new features, fix bugs, or improve the documentation, feel free to open an issue or submit a pull request.

---

## final thoughts

ai doesn't have to be intimidating. with the right tools, it can be fun, interactive, and even a little magical. this playground is my way of making machine learning more accessible and enjoyable. so go ahead, play around, tweak some settings, and see what you can create. who knows? you might just build something amazing.

happy coding! 🚀

import * as tf from '@tensorflow/tfjs';

/**
 * Creates a Multi-Layer Perceptron (MLP) model
 * MLPs are fully connected networks used for structured/tabular data.
 * @param {number} hiddenUnits - Number of units in the first hidden layer
 * @param {number} learningRate - Learning rate for optimization
 * @returns {tf.Model} - A compiled TensorFlow.js MLP model
 */
export function createMLP(hiddenUnits, learningRate) {
    const model = tf.sequential();

    model.add(tf.layers.dense({ units: hiddenUnits, activation: 'relu', inputShape: [10] }));
    model.add(tf.layers.batchNormalization()); // Normalizes activations
    model.add(tf.layers.dropout(0.2)); // Prevents overfitting

    model.add(tf.layers.dense({ units: Math.max(4, hiddenUnits / 2), activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // Binary classification output

    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

/**
 * Creates a Convolutional Neural Network (CNN) for image classification.
 * CNNs optimize spatial data learning through convolutional layers.
 * @param {number} filters - Number of convolutional filters
 * @param {number} learningRate - Learning rate for optimization
 * @returns {tf.Model} - A compiled TensorFlow.js CNN model
 */
export function createCNN(filters, learningRate) {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        filters: filters,
        kernelSize: 3,
        activation: 'relu',
        inputShape: [28, 28, 1], // Explicitly define input shape
        padding: 'same' // Prevents dimensionality issues
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] })); // Downsampling
    model.add(tf.layers.dropout(0.3));

    model.add(tf.layers.flatten()); // Convert 2D feature maps into 1D vector
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout(0.3)); 
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' })); // 10-class output

    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'categoricalCrossentropy', // Ensure multi-class classification
        metrics: ['accuracy']
    });

    return model;
}


/**
 * Creates a Simple Recurrent Neural Network (RNN)
 * RNNs process sequential data by maintaining internal state.
 * @param {number} hiddenUnits - Number of hidden units in the RNN layer
 * @param {number} learningRate - Learning rate for optimization
 * @returns {tf.Model} - A compiled TensorFlow.js RNN model
 */
export function createRNN(hiddenUnits, learningRate) {
    const model = tf.sequential();

    model.add(tf.layers.simpleRNN({ 
        units: hiddenUnits, activation: 'relu', inputShape: [20, 10], returnSequences: true
    }));
    model.add(tf.layers.dropout(0.2)); // Helps prevent overfitting
    model.add(tf.layers.simpleRNN({ 
        units: Math.max(4, hiddenUnits / 2), activation: 'relu'
    }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // Binary classification

    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

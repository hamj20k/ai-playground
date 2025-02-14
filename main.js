import * as tf from '@tensorflow/tfjs';
import { createMLP, createCNN, createRNN } from './model.js';
import { initializeCharts, updateGraphs } from './graph.js';
import { initializeNetwork, updateNetwork } from './network.js';

// Store charts globally within the module
let charts;

document.addEventListener("DOMContentLoaded", async function () {
    console.log("🚀 DOM fully loaded, initializing...");

/**
 * Updates slider values dynamically when changed
 */
    function updateSliderValue(event) {
        const { id, value } = event.target;
        const valueSpan = document.getElementById(`${id}Value`);
        if (valueSpan) {
            valueSpan.innerText = value;
        }
    }

    /**
     * Initializes sliders and ensures correct default values are displayed
     */
    function initializeSliders() {
        const sliders = {
            learningRate: 0.01,
            batchSize: 16,
            epochs: 10,
            hiddenUnits: 32,
            filters: 8
        };

        // Set default values and attach event listeners
        Object.entries(sliders).forEach(([id, defaultValue]) => {
            const slider = document.getElementById(id);
            const valueSpan = document.getElementById(`${id}Value`);

            if (slider && valueSpan) {
                valueSpan.innerText = slider.value || defaultValue; // Ensure correct value display
                slider.addEventListener("input", updateSliderValue);
            }
        });
    }

    // Call function after DOM is fully loaded
    document.addEventListener("DOMContentLoaded", initializeSliders);
    

    // Initialize all required DOM elements
    const elements = {
        mlpBtn: document.getElementById("mlpBtn"),
        cnnBtn: document.getElementById("cnnBtn"),
        rnnBtn: document.getElementById("rnnBtn"),
        startTrainingBtn: document.getElementById("startTraining"),
        logsContainer: document.getElementById("logs"),
        consoleContainer: document.getElementById("console"),
        mlpParams: document.getElementById("mlpParams"),
        cnnParams: document.getElementById("cnnParams"),
        sliders: document.querySelectorAll("input[type='range']")
    };

    // Validate all elements exist
    for (const [key, element] of Object.entries(elements)) {
        if (!element) {
            console.error(`❌ Element not found: ${key}`);
            return;
        }
    }

    // Initialize state
    let state = {
        selectedModel: "mlp",
        isTraining: false,
        model: null
    };

    // Initialize charts safely
    try {
        charts = initializeCharts();
        console.log("✅ Charts initialized successfully");
    } catch (error) {
        console.error("❌ Failed to initialize charts:", error);
        return;
    }

    /**
     * Updates UI to show relevant parameters for selected model
     */
    function showParams(modelType) {
        state.selectedModel = modelType;
        console.log(`✅ Model selected: ${modelType}`);

        // Update parameter visibility
        elements.mlpParams.style.display = modelType === "mlp" ? "block" : "none";
        elements.cnnParams.style.display = modelType === "cnn" ? "block" : "none";

        // Update button states
        [elements.mlpBtn, elements.cnnBtn, elements.rnnBtn].forEach(btn => {
            btn.classList.remove("selected");
        });
        elements[`${modelType}Btn`].classList.add("selected");

        // Initialize network visualization for new model
        initializeNetwork(modelType, getHyperparameters());
        updateModelInfo(modelType);
    }

    /**
     * Updates displayed values for sliders
     */
    function updateSliderValue(event) {
        const valueSpan = document.getElementById(`${event.target.id}Value`);
        if (valueSpan) {
            valueSpan.innerText = event.target.value;
            console.log(`🔄 ${event.target.id} updated to ${event.target.value}`);
        }
    }

    /**
     * Initializes all sliders with default values
     */
    function initializeSliders() {
        const defaults = {
            learningRate: 0.01,
            batchSize: 16,
            epochs: 10,
            hiddenUnits: 32,
            filters: 8
        };

        Object.entries(defaults).forEach(([id, value]) => {
            const slider = document.getElementById(id);
            const valueSpan = document.getElementById(`${id}Value`);
            
            if (slider && valueSpan) {
                slider.value = value;
                valueSpan.innerText = value;
                slider.addEventListener("input", updateSliderValue);
            }
        });
    }

    /**
     * Collects current hyperparameter values
     */
    function getHyperparameters() {
        return {
            learningRate: parseFloat(document.getElementById("learningRate").value),
            batchSize: parseInt(document.getElementById("batchSize").value),
            epochs: parseInt(document.getElementById("epochs").value),
            hiddenUnits: parseInt(document.getElementById("hiddenUnits").value),
            filters: parseInt(document.getElementById("filters").value)
        };
    }

    /**
 * Dynamically updates the model information and UI sections.
 * Ensures smooth transitions and structured information display.
 *  @param {string} modelType - The selected model type ("mlp", "cnn", or "rnn")
 */
    function updateModelInfo(modelType) {
        const modelTitle = document.getElementById("modelTitle");
        const modelDescription = document.getElementById("modelDescription");
        const modelDetails = document.getElementById("modelDetails");
        const modelParamsContainer = document.getElementById("modelParamsContainer");
        const presetParamsList = document.getElementById("presetParams");
        const variableParamsList = document.getElementById("variableParams");
        const modelInfoContainer = document.getElementById("modelInfoContainer");
    
        const toggleModelInfo = document.getElementById("toggleModelInfo");
        const toggleModelParams = document.getElementById("toggleModelParams");
    
        const descriptions = {
            mlp: {
                title: "Multi-Layer Perceptron (MLP)",
                description: "A fully connected neural network suitable for tabular and structured data classification. MLP processes flattened inputs using dense layers and is effective for simple classification tasks like digit recognition.",
                details: [
                    "<b>Fully connected layers:</b> Each neuron is connected to every neuron in the next layer.",
                    "<b>Flattens input images:</b> Converts 2D images into a 1D vector before processing.",
                    "<b>ReLU activation:</b> Helps the network capture complex patterns by introducing non-linearity.",
                    "<b>Batch normalization:</b> Stabilizes training by normalizing activations.",
                    "<b>Dropout layers:</b> Prevents overfitting by randomly disabling neurons during training.",
                    "<b>Best for:</b> Structured/tabular data and simple image classification.",
                    "<b>Downside:</b> Lacks spatial awareness, making it less effective than CNNs for image recognition."
                ],
                parameters: {
                    preset: [
                        "<b>Input Shape:</b> 1D array of length 10",
                        "<b>Final Layer Activation:</b> Softmax (for multi-class classification)"
                    ],
                    variable: [
                        "<b>Hidden Units:</b> Adjusts model complexity.",
                        "<b>Learning Rate:</b> Controls step size during optimization.",
                        "<b>Batch Size:</b> Determines how many samples are processed at a time.",
                        "<b>Epochs:</b> Number of training cycles."
                    ]
                }
            },
    
            cnn: {
                title: "Convolutional Neural Network (CNN)",
                description: "A specialized neural network for image recognition and feature extraction. CNNs process input images by learning spatial hierarchies using convolutional and pooling layers.",
                details: [
                    "<b>Convolutional layers:</b> Detect spatial patterns like edges, textures, and shapes.",
                    "<b>Max-pooling layers:</b> Reduce dimensionality while retaining key features.",
                    "<b>Batch normalization:</b> Speeds up training and improves stability.",
                    "<b>Dropout layers:</b> Prevent overfitting by randomly disabling neurons.",
                    "<b>Best for:</b> Image processing, object recognition, and feature extraction.",
                    "<b>Downside:</b> More computationally intensive than MLPs."
                ],
                parameters: {
                    preset: [
                        "<b>Input Shape:</b> 28×28×1 (Grayscale Image)",
                        "<b>Final Layer Activation:</b> Softmax (for multi-class classification)",
                        "<b>Kernel Size:</b> 3×3 (Convolution Window)"
                    ],
                    variable: [
                        "<b>Number of Filters:</b> Determines how many feature detectors are used.",
                        "<b>Learning Rate:</b> Controls optimization step size.",
                        "<b>Batch Size:</b> Affects memory usage and training speed.",
                        "<b>Epochs:</b> Number of training iterations.",
                        "<b>Pooling Size:</b> Defines how much downsampling is applied."
                    ]
                }
            },
    
            rnn: {
                title: "Recurrent Neural Network (RNN)",
                description: "A sequence-based model that processes time-series or sequential data by maintaining a hidden state that captures past information.",
                details: [
                    "<b>Processes data sequentially:</b> Remembers past information through a hidden state.",
                    "<b>Simple RNN layers:</b> Capture time dependencies in data.",
                    "<b>Useful for:</b> Speech recognition, stock price prediction, and text analysis.",
                    "<b>Downside:</b> Struggles with long-term dependencies (vanishing gradient problem).",
                    "<b>Best for:</b> Sequential data like time-series, speech, and language models."
                ],
                parameters: {
                    preset: [
                        "<b>Input Shape:</b> 28×28 (Sequential Data Representation)",
                        "<b>Final Layer Activation:</b> Softmax (for multi-class classification)"
                    ],
                    variable: [
                        "<b>Hidden Units:</b> Determines how much past information is retained.",
                        "<b>Learning Rate:</b> Controls update speed during training.",
                        "<b>Batch Size:</b> Number of sequences processed at once.",
                        "<b>Epochs:</b> Number of training cycles.",
                    ]
                }
            }
        };
    
        // Update model title and description
        modelTitle.innerText = descriptions[modelType].title;
        modelDescription.innerHTML = descriptions[modelType].description;
    
        // Update details list
        modelDetails.innerHTML = "";
        descriptions[modelType].details.forEach(detail => {
            let li = document.createElement("li");
            li.innerHTML = detail;
            modelDetails.appendChild(li);
        });
    
        // Update parameters section
        presetParamsList.innerHTML = "";
        variableParamsList.innerHTML = "";
    
        descriptions[modelType].parameters.preset.forEach(param => {
            let li = document.createElement("li");
            li.innerHTML = param;
            presetParamsList.appendChild(li);
        });
    
        descriptions[modelType].parameters.variable.forEach(param => {
            let li = document.createElement("li");
            li.innerHTML = param;
            variableParamsList.appendChild(li);
        });
    
        // Show sections
        modelInfoContainer.style.display = "block";
        modelParamsContainer.style.display = "block";
    
        // Toggle Model Info
        toggleModelInfo.onclick = () => {
            if (modelInfoContainer.style.display === "none") {
                modelInfoContainer.style.display = "block";
                toggleModelInfo.innerText = "Hide Model Info ⬆";
            } else {
                modelInfoContainer.style.display = "none";
                toggleModelInfo.innerText = "Show Model Info ⬇";
            }
        };
    
        // Toggle Model Parameters
        toggleModelParams.onclick = () => {
            if (modelParamsContainer.style.display === "none") {
                modelParamsContainer.style.display = "block";
                toggleModelParams.innerText = "Hide Model Parameters ⬆";
            } else {
                modelParamsContainer.style.display = "none";
                toggleModelParams.innerText = "Show Model Parameters ⬇";
            }
        };
    }
    

    /**
     * Adds a message to the logs
     */
    function logMessage(message) {
        elements.logsContainer.innerHTML += `<p>> ${message}</p>`;
        elements.logsContainer.scrollTop = elements.logsContainer.scrollHeight;
    }

    /**
     * Handles the training process
     */
    async function trainModel() {
        if (state.isTraining) {
            state.isTraining = false;
            elements.startTrainingBtn.innerText = "Start Training";
            elements.consoleContainer.classList.remove("training-active");
            logMessage("Training stopped.");
            console.log("🚫 Training stopped by user.");
            return;
        }

        // Initialize training state
        state.isTraining = true;
        elements.startTrainingBtn.innerText = "Stop Training";
        elements.consoleContainer.classList.add("training-active");
        elements.logsContainer.innerHTML = "";
        logMessage("Model initialized.");
        logMessage(`Training ${state.selectedModel.toUpperCase()} model...`);

        const params = getHyperparameters();

        // Cleanup previous model
        if (state.model) {
            state.model.dispose();
            console.log("♻️ Previous model disposed.");
        }

        // Create new model
        try {
            state.model = state.selectedModel === "mlp"
                ? createMLP(params.hiddenUnits, params.learningRate)
                : state.selectedModel === "cnn"
                    ? createCNN(params.filters, params.learningRate)
                    : createRNN(params.hiddenUnits, params.learningRate);
            
            console.log("✅ New model created successfully");
        } catch (error) {
            console.error("❌ Failed to create model:", error);
            logMessage("Error creating model.");
            return;
        }

        // Generate training data
        const [xs, ys] = tf.tidy(() => {
            const sampleCount = 100;
            
            const xData = state.selectedModel === "cnn"
                ? tf.randomNormal([sampleCount, 9, 9, 1], 0, 1)
                : state.selectedModel === "rnn"
                    ? tf.randomNormal([sampleCount, 20, 10])
                    : tf.randomNormal([sampleCount, 10]);

            const yData = state.selectedModel === "cnn"
                ? tf.oneHot(tf.floor(tf.randomUniform([sampleCount], 0, 10)), 10)
                : tf.round(tf.randomUniform([sampleCount, 1]));

            return [xData, yData];
        });

        // Training loop
        try {
            for (let epoch = 0; epoch < params.epochs; epoch++) {
                if (!state.isTraining) {
                    logMessage("Training interrupted.");
                    break;
                }

                const history = await state.model.fit(xs, ys, {
                    epochs: 1,
                    batchSize: params.batchSize
                });

                const loss = history.history.loss[0].toFixed(4);
                const accuracy = history.history.accuracy
                    ? (history.history.accuracy[0] * 100).toFixed(2)
                    : history.history.acc
                        ? (history.history.acc[0] * 100).toFixed(2)
                        : "N/A"; 

                logMessage(`Epoch ${epoch + 1}: loss = ${loss}, accuracy = ${accuracy}%`);
                updateGraphs(epoch + 1, loss, accuracy);

                // Update network visualization
                const weights = {};
                state.model.layers.forEach((layer, i) => {
                    if (layer.getWeights().length > 0) {
                        const weightTensor = layer.getWeights()[0];
                        const weightData = weightTensor.dataSync();
                        weightData.forEach((value, j) => {
                            weights[`${i}-${j}`] = value.toFixed(2);
                        });
                    }
                });

                if (Object.keys(weights).length > 0) {
                    updateNetwork(weights);
                }

                await tf.nextFrame();
            }
        } catch (error) {
            console.error("❌ Training error:", error);
            logMessage("Error during training.");
        } finally {
            tf.dispose([xs, ys]);
            state.isTraining = false;
            elements.startTrainingBtn.innerText = "Start Training";
            elements.consoleContainer.classList.remove("training-active");
            logMessage("Training complete.");
        }
    }

    showParams("mlp");
    initializeSliders();

    elements.mlpBtn.addEventListener("click", () => showParams("mlp"));
    elements.cnnBtn.addEventListener("click", () => showParams("cnn"));
    elements.rnnBtn.addEventListener("click", () => showParams("rnn"));
    elements.startTrainingBtn.addEventListener("click", trainModel);
});

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
                ? tf.randomNormal([sampleCount, 28, 28, 1])
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


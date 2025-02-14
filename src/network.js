// Colors for network visualization
const COLORS = {
    background: "#000000",
    neuron: "#33ff33",
    connection: "#22aa22",
    weightText: "#ffcc00",
    errorText: "#ff0000"
};

// Global state for the network visualization
let state = {
    canvas: null,
    ctx: null,
    layers: [],
    weights: {},
    selectedModel: null,
    isInitialized: false
};

/**
 * Initializes the network visualization
 * @param {string} model - The selected model type
 * @param {Object} params - Model parameters
 */
export function initializeNetwork(model, params) {
    console.log(`🖥️ Initializing Network for ${model}...`);

    if (!state.isInitialized) {
        const canvas = document.getElementById("networkCanvas");
        if (!canvas) {
            console.error("❌ Network canvas not found.");
            return;
        }

        state.canvas = canvas;
        state.ctx = canvas.getContext("2d");

        // Set fixed dimensions
        canvas.width = 600;
        canvas.height = 300;

        state.isInitialized = true;
    }

    setNetworkStructure(model, params);
    requestAnimationFrame(drawNetwork);
}

/**
 * Sets the network structure based on model type
 * @param {string} model - The selected model type
 * @param {Object} params - Model parameters
 */
function setNetworkStructure(model, params) {
    state.selectedModel = model;

    switch (model) {
        case "mlp":
            state.layers = [10, params.hiddenUnits, Math.max(4, params.hiddenUnits / 2), 1];
            break;
        case "cnn":
            state.layers = [28 * 28, params.filters * 4, 64, 10];
            break;
        case "rnn":
            state.layers = [20, params.hiddenUnits, Math.floor(params.hiddenUnits / 2), 1];
            break;
        default:
            state.layers = [];
            console.error("❌ Unknown model type:", model);
    }
}

/**
 * Draws the neural network visualization
 */
function drawNetwork() {
    if (!state.ctx || state.layers.length === 0) {
        console.warn("⚠️ Cannot draw network - missing context or layers.");
        return;
    }

    const { ctx, canvas, layers, weights } = state;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const xSpacing = canvas.width / (layers.length + 1);
    ctx.fillStyle = COLORS.neuron;
    ctx.strokeStyle = COLORS.connection;

    layers.forEach((neurons, layerIndex) => {
        const ySpacing = canvas.height / (neurons + 1);

        for (let neuronIndex = 0; neuronIndex < neurons; neuronIndex++) {
            const x = xSpacing * (layerIndex + 1);
            const y = ySpacing * (neuronIndex + 1);

            // Draw neuron
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fill();

            if (layerIndex > 0) {
                const prevLayerSize = layers[layerIndex - 1];
                for (let prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                    const prevX = xSpacing * layerIndex;
                    const prevY = (canvas.height / (prevLayerSize + 1)) * (prevNeuron + 1);

                    // Draw connection
                    ctx.beginPath();
                    ctx.moveTo(prevX, prevY);
                    ctx.lineTo(x, y);
                    ctx.stroke();
                }
            }
        }
    });
}

/**
 * Updates the network visualization with new weights
 * @param {Object} newWeights - The updated weight values
 */
export function updateNetwork(newWeights) {
    console.log("🔄 Updating Network with New Weights...");
    state.weights = newWeights;
    requestAnimationFrame(drawNetwork);
}

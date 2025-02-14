import { initializeCharts } from "./graph";

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
    isInitialized: false,
    nodePositions: []
};

/**
 * Initializes the network visualization
 * @param {string} model - The selected model type
 * @param {Object} params - Model parameters
 */
export function initializeNetwork(model, params) {
    console.log(`🖥️ Initializing Network for ${model}...`);

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

    initializeCharts(); // Ensure charts are initialized before training
    setNetworkStructure(model, params);
    drawNetwork(); // Draw the network immediately
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
            return;
    }

    computeNodePositions();
}

/**
 * Computes the positions of neurons in each layer for smooth rendering.
 */
function computeNodePositions() {
    const { canvas, layers } = state;
    state.nodePositions = [];

    const xSpacing = canvas.width / (layers.length + 1);

    layers.forEach((neurons, layerIndex) => {
        const ySpacing = canvas.height / (neurons + 1);
        for (let neuronIndex = 0; neuronIndex < neurons; neuronIndex++) {
            const x = xSpacing * (layerIndex + 1);
            const y = ySpacing * (neuronIndex + 1);
            state.nodePositions.push({ x, y, layer: layerIndex, index: neuronIndex });
        }
    });
}

/**
 * Draws the neural network visualization.
 */
function drawNetwork() {
    if (!state.ctx || state.layers.length === 0) {
        console.warn("⚠️ Cannot draw network - missing context or layers.");
        return;
    }

    const { ctx, canvas, nodePositions, weights } = state;
    
    // Clear canvas before drawing
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connections (weights)
    nodePositions.forEach(({ x, y, layer, index }) => {
        if (layer > 0) {
            const prevLayerNodes = nodePositions.filter(node => node.layer === layer - 1);
            prevLayerNodes.forEach(({ x: prevX, y: prevY, index: prevIndex }) => {
                const weightKey = `${layer - 1}-${prevIndex}-${index}`;
                const weightValue = weights[weightKey] || (Math.random() * 2 - 1).toFixed(2);

                ctx.strokeStyle = COLORS.connection;
                ctx.lineWidth = Math.abs(weightValue) * 2; // Thicker line for stronger weights
                ctx.globalAlpha = Math.min(1, Math.abs(weightValue)); // Adjust opacity based on strength
                
                ctx.beginPath();
                ctx.moveTo(prevX, prevY);
                ctx.lineTo(x, y);
                ctx.stroke();

                ctx.globalAlpha = 1.0; // Reset opacity

                // Draw weight text
                ctx.fillStyle = COLORS.weightText;
                ctx.font = "10px Courier New";
                ctx.fillText(weightValue, (prevX + x) / 2 + 5, (prevY + y) / 2 + 5);
            });
        }
    });

    // Draw neurons
    nodePositions.forEach(({ x, y }) => {
        ctx.fillStyle = COLORS.neuron;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
    });

    console.log("🖥️ Network visualization updated.");
}

/**
 * Updates the network visualization with new weights.
 * @param {Object} newWeights - The updated weight values
 */
export function updateNetwork(newWeights) {
    console.log("🔄 Updating Network with New Weights...");
    state.weights = newWeights;
    drawNetwork();
}

/**
 * Resets the network visualization.
 */
export function resetNetwork() {
    console.log("🔄 Resetting Network...");
    state.weights = {};
    drawNetwork();
}

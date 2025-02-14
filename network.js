import { initializeCharts } from "./graph";

// Colors for visualization
const COLORS = {
    background: "#000000",
    neuron: "#33ff33",
    hiddenNeuron: "#888888",
    weightWeak: "#55aa55",  // Light green - Weak connection
    weightMedium: "#33cc33", // Medium green - Medium connection
    weightStrong: "#22ff22", // Bright green - Strong connection
};

// Global state
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
 * Initializes the neural network visualization.
 * @param {string} model - The selected model type
 * @param {Object} params - Model parameters (hidden units, filters, etc.)
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

    // Set optimized dimensions for better visibility
    canvas.width = 900;
    canvas.height = 500;

    state.isInitialized = true;

    initializeCharts(); // Ensure charts are initialized before training
    setNetworkStructure(model, params);
    drawNetwork();

    // Attach event listener to update graph on slider change
    document.getElementById("hiddenUnits").addEventListener("input", (event) => {
        updateNetworkStructure(model, parseInt(event.target.value));
    });
}

/**
 * Sets the network structure dynamically based on the model and slider values.
 */
function setNetworkStructure(model, params) {
    state.selectedModel = model;
    updateNetworkStructure(model, params.hiddenUnits);
}

/**
 * Updates network structure when the **hidden units slider** changes.
 */
function updateNetworkStructure(model, hiddenUnits) {
    console.log(`🔄 Updating hidden layer neurons: ${hiddenUnits}`);

    switch (model) {
        case "mlp":
            state.layers = [10, hiddenUnits, Math.max(4, hiddenUnits / 2), 1];
            break;
        case "cnn":
            state.layers = [81, hiddenUnits, 128, 10]; // Hidden units affect second layer
            break;
        case "rnn":
            state.layers = [20, hiddenUnits, Math.max(4, hiddenUnits / 2), 1];
            break;
        default:
            state.layers = [];
            console.error("❌ Unknown model type:", model);
            return;
    }

    computeNodePositions();
    drawNetwork();
}

/**
 * Computes neuron positions for proper layout.
 */
function computeNodePositions() {
    const { canvas, layers } = state;
    state.nodePositions = [];

    const totalLayers = layers.length;
    const xSpacing = canvas.width / (totalLayers + 1); // Evenly space layers horizontally

    layers.forEach((neurons, layerIndex) => {
        let ySpacing = canvas.height / (neurons + 1);

        for (let neuronIndex = 0; neuronIndex < neurons; neuronIndex++) {
            const x = xSpacing * (layerIndex + 1);
            const y = ySpacing * (neuronIndex + 1);

            state.nodePositions.push({ x, y, layer: layerIndex, index: neuronIndex });
        }
    });
}

/**
 * Draws the network visualization.
 */
function drawNetwork() {
    if (!state.ctx || state.layers.length === 0) {
        console.warn("⚠️ Cannot draw network - missing context or layers.");
        return;
    }

    const { ctx, canvas, nodePositions, weights } = state;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connections (weights)
    nodePositions.forEach(({ x, y, layer, index }) => {
        if (layer > 0) {
            const prevLayerNodes = nodePositions.filter(node => node.layer === layer - 1);
            prevLayerNodes.forEach(({ x: prevX, y: prevY, index: prevIndex }) => {
                const weightKey = `${layer - 1}-${prevIndex}-${index}`;
                const weightValue = weights[weightKey] || (Math.random() * 2 - 1).toFixed(2);
                const absWeight = Math.abs(weightValue);

                // Assign colors based on weight strength
                let connectionColor;
                if (absWeight < 0.3) connectionColor = COLORS.weightWeak;  // Weak
                else if (absWeight < 0.7) connectionColor = COLORS.weightMedium; // Medium
                else connectionColor = COLORS.weightStrong; // Strong

                ctx.strokeStyle = connectionColor;
                ctx.lineWidth = absWeight * 2;

                ctx.beginPath();
                ctx.moveTo(prevX, prevY);
                ctx.lineTo(x, y);
                ctx.stroke();
            });
        }
    });

    // Draw neurons
    nodePositions.forEach(({ x, y, layer }) => {
        ctx.fillStyle = layer === 0 || layer === state.layers.length - 1 ? COLORS.neuron : COLORS.hiddenNeuron;
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.fill();
    });

    console.log("🖥️ Network visualization updated.");
}

/**
 * Updates network visualization with new weights.
 */
export function updateNetwork(newWeights) {
    state.weights = newWeights;
    drawNetwork();
}

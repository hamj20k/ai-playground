import Chart from 'chart.js/auto';

// Store references to both charts
let lossChart, accuracyChart;

/**
 * Initializes the training charts (Loss & Accuracy)
 * @returns {Object} References to the loss and accuracy charts
 */
export function initializeCharts() {
    console.log("📊 Initializing training charts...");

    const lossCanvas = document.getElementById("lossChart");
    const accuracyCanvas = document.getElementById("accuracyChart");

    if (!lossCanvas || !accuracyCanvas) {
        console.error("❌ Chart canvas elements not found. Check your HTML structure.");
        return null;
    }

    // Destroy existing charts if they already exist
    if (lossChart) lossChart.destroy();
    if (accuracyChart) accuracyChart.destroy();

    // Maximum epochs to display
    const MAX_EPOCHS_SHOWN = 20;

    // Initialize Loss Chart
    lossChart = new Chart(lossCanvas.getContext("2d"), {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Loss",
                data: [],
                borderColor: "#33ff33", // Neon Green
                borderWidth: 2,
                fill: false,
                pointRadius: 3,
                pointBackgroundColor: "#33ff33",
                tension: 0.2 // Smooth curves
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false, // Real-time update
            scales: {
                x: { grid: { color: "#555" }, ticks: { color: "#33ff33" } },
                y: { grid: { color: "#555" }, ticks: { color: "#33ff33" } }
            },
            plugins: { legend: { labels: { color: "#33ff33" } } }
        }
    });

    // Initialize Accuracy Chart
    accuracyChart = new Chart(accuracyCanvas.getContext("2d"), {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Accuracy",
                data: [],
                borderColor: "#ffcc00", // Yellow
                borderWidth: 2,
                fill: false,
                pointRadius: 3,
                pointBackgroundColor: "#ffcc00",
                tension: 0.2 // Smooth curves
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false, // Real-time update
            scales: {
                x: { grid: { color: "#555" }, ticks: { color: "#ffcc00" } },
                y: { grid: { color: "#555" }, ticks: { color: "#ffcc00" } }
            },
            plugins: { legend: { labels: { color: "#ffcc00" } } }
        }
    });

    return { lossChart, accuracyChart };
}

/**
 * Updates the charts with new training data
 * @param {number} epoch - Current epoch number
 * @param {number} loss - Current loss value
 * @param {number} accuracy - Current accuracy value
 */
export function updateGraphs(epoch, loss, accuracy) {
    if (!lossChart || !accuracyChart) {
        console.error("❌ Charts not initialized. Call `initializeCharts()` first.");
        return;
    }

    console.log(`📊 Updating graphs - Epoch ${epoch}: Loss = ${loss}, Accuracy = ${accuracy}%`);

    const MAX_EPOCHS_SHOWN = 20;

    // Maintain fixed number of epochs on charts
    if (lossChart.data.labels.length >= MAX_EPOCHS_SHOWN) {
        lossChart.data.labels.shift();
        lossChart.data.datasets[0].data.shift();
    }
    if (accuracyChart.data.labels.length >= MAX_EPOCHS_SHOWN) {
        accuracyChart.data.labels.shift();
        accuracyChart.data.datasets[0].data.shift();
    }

    // Add new data points
    lossChart.data.labels.push(`Epoch ${epoch}`);
    lossChart.data.datasets[0].data.push(loss);
    lossChart.update();

    accuracyChart.data.labels.push(`Epoch ${epoch}`);
    accuracyChart.data.datasets[0].data.push(accuracy);
    accuracyChart.update();
}

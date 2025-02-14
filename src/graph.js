import Chart from 'chart.js/auto';

// Global chart references
let lossChart, accuracyChart;

/**
 * Initializes the loss and accuracy charts.
 */
export function initializeCharts() {
    console.log("📊 Initializing training charts...");

    const lossCanvas = document.getElementById("lossChart");
    const accuracyCanvas = document.getElementById("accuracyChart");

    if (!lossCanvas || !accuracyCanvas) {
        console.error("❌ Chart canvas elements not found. Check your HTML structure.");
        return;
    }

    // Destroy existing charts to prevent duplicates
    if (lossChart) lossChart.destroy();
    if (accuracyChart) accuracyChart.destroy();

    lossChart = new Chart(lossCanvas, {
        type: "line",
        data: { labels: [], datasets: [{ label: "Loss", data: [], borderColor: "#ff3333", borderWidth: 2, pointBackgroundColor: "#ff3333", tension: 0.2 }] },
        options: { responsive: true, animation: false, maintainAspectRatio: false, scales: { x: { ticks: { color: "#ffffff" } }, y: { ticks: { color: "#ffffff" } } } }
    });

    accuracyChart = new Chart(accuracyCanvas, {
        type: "line",
        data: { labels: [], datasets: [{ label: "Accuracy", data: [], borderColor: "#33ff33", borderWidth: 2, pointBackgroundColor: "#33ff33", tension: 0.2 }] },
        options: { responsive: true, animation: false, maintainAspectRatio: false, scales: { x: { ticks: { color: "#ffffff" } }, y: { ticks: { color: "#ffffff" } } } }
    });

    return { lossChart, accuracyChart };
}

/**
 * Updates the loss and accuracy graphs dynamically.
 */
export function updateGraphs(epoch, loss, accuracy) {
    if (!lossChart || !accuracyChart) {
        console.error("❌ Charts not initialized. Call `initializeCharts()` first.");
        return;
    }

    console.log(`📊 Updating graphs - Epoch ${epoch}: Loss = ${loss}, Accuracy = ${accuracy}%`);

    if (!isNaN(loss)) {
        lossChart.data.labels.push(`Epoch ${epoch}`);
        lossChart.data.datasets[0].data.push(loss);
        lossChart.update();
    }

    if (!isNaN(accuracy)) {
        accuracyChart.data.labels.push(`Epoch ${epoch}`);
        accuracyChart.data.datasets[0].data.push(accuracy);
        accuracyChart.update();
    }
}

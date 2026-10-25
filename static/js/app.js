/**
 * Credit Card Fraud Detection - Interactive Frontend Logic
 * Handles real-time API communication, validation, dynamic result rendering,
 * session prediction history, and Chart.js benchmarks.
 */

// Global State
let sampleTransactions = [];
let comparisonChart = null;
let predictionHistory = [];
let sessionStats = {
  total: 0,
  frauds: 0,
  legits: 0
};

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  fetchHealthStatus();
  fetchModelMetrics();
  fetchSampleTransactions();
  setupEventListeners();
});

/**
 * Checks system health and model availability
 */
async function fetchHealthStatus() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const statusBadge = document.getElementById("system-status-badge");
    if (data.status === "ok" && data.model_loaded) {
      statusBadge.innerHTML = `<i class="fa-solid fa-circle-check me-1"></i> Model Online: ${data.model_name}`;
      statusBadge.className = "badge bg-success-subtle text-success border border-success-subtle px-2 py-1";
    } else {
      statusBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation me-1"></i> Model Not Loaded`;
      statusBadge.className = "badge bg-warning-subtle text-warning border border-warning-subtle px-2 py-1";
    }
  } catch (err) {
    console.error("Health check failed:", err);
  }
}

/**
 * Loads real metrics, updates KPI cards, populates comparison table, and renders Chart.js
 */
async function fetchModelMetrics() {
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();

    if (data.status !== "success") return;

    const tm = data.test_metrics || {};
    const comp = data.model_comparison || [];

    // Update KPIs
    if (tm.recall !== undefined) document.getElementById("kpi-recall").textContent = (tm.recall * 100).toFixed(2) + "%";
    if (tm.precision !== undefined) document.getElementById("kpi-precision").textContent = (tm.precision * 100).toFixed(2) + "%";
    if (tm.f1_score !== undefined) document.getElementById("kpi-f1").textContent = tm.f1_score.toFixed(4);
    if (tm.pr_auc !== undefined) document.getElementById("kpi-pr-auc").textContent = tm.pr_auc.toFixed(4);

    // Populate Comparison Table
    const tbody = document.getElementById("comparison-tbody");
    if (tbody && comp.length > 0) {
      tbody.innerHTML = "";
      comp.forEach((row, idx) => {
        const isBest = idx === 0;
        const tr = document.createElement("tr");
        if (isBest) tr.className = "table-primary fw-bold";
        
        tr.innerHTML = `
          <td>
            ${isBest ? '<i class="fa-solid fa-crown text-warning me-1"></i>' : ''}
            ${row.Model}
            ${isBest ? '<span class="badge bg-primary text-white ms-2" style="font-size: 0.7rem;">Selected Ensemble</span>' : ''}
          </td>
          <td>${(row.Recall * 100).toFixed(2)}%</td>
          <td>${(row.Precision * 100).toFixed(2)}%</td>
          <td class="text-primary">${row["F1-Score"].toFixed(4)}</td>
          <td>${row["PR-AUC"].toFixed(4)}</td>
          <td>${row["ROC-AUC"].toFixed(4)}</td>
          <td>${(row.Accuracy * 100).toFixed(2)}%</td>
          <td class="text-muted">${row["Training Time (s)"]}s</td>
        `;
        tbody.appendChild(tr);
      });

      // Render Chart.js
      renderComparisonChart(comp);
    }

  } catch (err) {
    console.error("Failed to load metrics:", err);
  }
}

/**
 * Renders Chart.js comparing model F1-Score & PR-AUC
 */
function renderComparisonChart(modelsData) {
  const ctx = document.getElementById("modelComparisonChart");
  if (!ctx) return;

  const labels = modelsData.map(m => m.Model);
  const f1Scores = modelsData.map(m => m["F1-Score"]);
  const prAucScores = modelsData.map(m => m["PR-AUC"]);

  if (comparisonChart) {
    comparisonChart.destroy();
  }

  comparisonChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Validation F1-Score",
          data: f1Scores,
          backgroundColor: "#2563eb",
          borderRadius: 4,
        },
        {
          label: "Validation PR-AUC",
          data: prAucScores,
          backgroundColor: "#93c5fd",
          borderRadius: 4,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 1.0,
          grid: { color: "#f1f5f9" }
        },
        x: {
          grid: { display: false },
          ticks: {
            font: { size: 10 },
            maxRotation: 45,
            minRotation: 20
          }
        }
      },
      plugins: {
        legend: {
          position: "top",
          labels: { font: { size: 11, family: "Inter" } }
        },
        tooltip: {
          padding: 10,
          boxPadding: 4
        }
      }
    }
  });
}

/**
 * Fetches benchmark genuine sample transactions from dataset
 */
async function fetchSampleTransactions() {
  try {
    const res = await fetch("/api/sample-transactions");
    const data = await res.json();
    if (data.status === "success" && data.samples) {
      sampleTransactions = data.samples;
    }
  } catch (err) {
    console.error("Failed to load samples:", err);
  }
}

/**
 * Loads a sample into the form by index
 */
window.loadSampleByIndex = function(idx) {
  if (!sampleTransactions || idx >= sampleTransactions.length) return;
  const sample = sampleTransactions[idx];

  document.getElementById("amt").value = sample.amt;
  document.getElementById("category").value = sample.category;
  document.getElementById("hour").value = sample.hour;
  document.getElementById("day_of_week").value = sample.day_of_week;
  document.getElementById("distance_km").value = sample.distance_km;
  document.getElementById("age").value = sample.age;
  document.getElementById("gender").value = sample.gender;
  document.getElementById("city_pop").value = sample.city_pop;

  // Flash the form briefly to show it loaded
  const form = document.getElementById("prediction-form");
  form.style.transition = "background-color 0.3s";
  form.style.backgroundColor = "#eff6ff";
  setTimeout(() => {
    form.style.backgroundColor = "";
  }, 400);
};

/**
 * Event Listeners Setup
 */
function setupEventListeners() {
  const form = document.getElementById("prediction-form");
  const resetBtn = document.getElementById("btn-reset");

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      await handlePredictionSubmit();
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      form.reset();
      document.getElementById("result-empty-state").classList.remove("d-none");
      document.getElementById("result-card").classList.add("d-none");
      document.getElementById("result-timestamp").textContent = "Reset";
    });
  }
}

/**
 * Handles Prediction Form Submission
 */
async function handlePredictionSubmit() {
  const form = document.getElementById("prediction-form");
  const spinner = document.getElementById("predict-spinner");
  const submitBtn = document.getElementById("btn-predict");

  // Client validation
  const amt = parseFloat(document.getElementById("amt").value);
  if (isNaN(amt) || amt <= 0) {
    alert("Please enter a valid positive transaction amount.");
    document.getElementById("amt").focus();
    return;
  }

  const payload = {
    amt: amt,
    category: document.getElementById("category").value,
    hour: parseInt(document.getElementById("hour").value, 10),
    day_of_week: parseInt(document.getElementById("day_of_week").value, 10),
    distance_km: parseFloat(document.getElementById("distance_km").value),
    age: parseInt(document.getElementById("age").value, 10),
    gender: document.getElementById("gender").value,
    city_pop: parseFloat(document.getElementById("city_pop").value)
  };

  // UI loading state
  spinner.classList.remove("d-none");
  submitBtn.disabled = true;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const result = await res.json();

    if (result.status === "error") {
      alert("Prediction Error: " + result.message);
      return;
    }

    // Render result card
    renderPredictionResult(result, payload);

    // Append to session history
    addToHistory(result, payload);

  } catch (err) {
    console.error("Prediction request failed:", err);
    alert("Network or server error occurred during prediction.");
  } finally {
    spinner.classList.add("d-none");
    submitBtn.disabled = false;
  }
}

/**
 * Renders Prediction Outcome into the UI
 */
function renderPredictionResult(result, input) {
  document.getElementById("result-empty-state").classList.add("d-none");
  const card = document.getElementById("result-card");
  card.classList.remove("d-none");

  const badge = document.getElementById("result-status-badge");
  const riskLevel = document.getElementById("result-risk-level");
  const fraudProbText = document.getElementById("result-fraud-prob");
  const legitProbText = document.getElementById("result-legit-prob");
  const meterFill = document.getElementById("result-meter-fill");
  const thresholdVal = document.getElementById("result-threshold-val");
  const explanation = document.getElementById("result-explanation");
  const modelName = document.getElementById("result-model-name");
  const timestamp = document.getElementById("result-timestamp");

  const isFraud = result.prediction === 1;
  const fraudProbPct = (result.fraud_probability * 100).toFixed(1) + "%";
  const legitProbPct = (result.legit_probability * 100).toFixed(1) + "%";

  fraudProbText.textContent = fraudProbPct;
  legitProbText.textContent = legitProbPct;
  thresholdVal.textContent = result.threshold;
  explanation.textContent = result.explanation;
  modelName.textContent = result.model_name;
  timestamp.textContent = new Date().toLocaleTimeString();

  meterFill.style.width = Math.min(100, Math.max(2, result.fraud_probability * 100)) + "%";

  if (isFraud) {
    badge.className = "result-badge fraud";
    badge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span>Fraud Alert</span>`;
    riskLevel.className = "badge bg-danger-subtle text-danger border border-danger-subtle px-2 py-1";
    riskLevel.textContent = result.risk_level || "High Risk";
    meterFill.className = "risk-meter-fill high";
  } else {
    badge.className = "result-badge legit";
    badge.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span>Legitimate Transaction</span>`;
    riskLevel.className = "badge bg-success-subtle text-success border border-success-subtle px-2 py-1";
    riskLevel.textContent = result.risk_level || "Low Risk";
    meterFill.className = result.fraud_probability > 0.3 ? "risk-meter-fill moderate" : "risk-meter-fill low";
  }
}

/**
 * Appends transaction to session history table and updates summary
 */
function addToHistory(result, input) {
  const isFraud = result.prediction === 1;
  sessionStats.total += 1;
  if (isFraud) sessionStats.frauds += 1;
  else sessionStats.legits += 1;

  document.getElementById("summary-total").textContent = sessionStats.total;
  document.getElementById("summary-frauds").textContent = sessionStats.frauds;
  document.getElementById("summary-legit").textContent = sessionStats.legits;

  const tbody = document.getElementById("history-tbody");
  if (sessionStats.total === 1) {
    tbody.innerHTML = ""; // Clear empty state
  }

  const tr = document.createElement("tr");
  const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  tr.innerHTML = `
    <td>${sessionStats.total}</td>
    <td>${timeStr}</td>
    <td class="fw-bold">$${input.amt.toFixed(2)}</td>
    <td><span class="badge bg-light text-dark border">${input.category}</span></td>
    <td>${input.hour}:00</td>
    <td><strong>${(result.fraud_probability * 100).toFixed(1)}%</strong></td>
    <td>
      <span class="badge ${isFraud ? 'bg-danger-subtle text-danger' : 'bg-success-subtle text-success'}">
        ${result.risk_level}
      </span>
    </td>
    <td>
      <span class="badge ${isFraud ? 'bg-danger text-white' : 'bg-success text-white'} px-2 py-1">
        ${isFraud ? 'FRAUD' : 'LEGIT'}
      </span>
    </td>
  `;

  // Prepend row to top
  tbody.insertBefore(tr, tbody.firstChild);
}

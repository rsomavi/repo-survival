const API_URL = "http://localhost:5000";

const form = document.getElementById("predict-form");
const btn = document.getElementById("predict-btn");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const repoUrl = document.getElementById("repo-url").value.trim();
  resultBox.classList.add("hidden");
  errorBox.classList.add("hidden");
  btn.disabled = true;
  btn.textContent = "Checking...";

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_url: repoUrl }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Something went wrong");
    }

    renderResult(data);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Check pulse";
  }
});

function renderResult(data) {
  const pct = Math.round(data.survival_probability * 100);
  const isActive = data.prediction === "Active";

  resultBox.innerHTML = `
    <div class="result-verdict">
      <span class="result-prob">${pct}%</span>
      <span class="result-tag ${isActive ? "active" : "abandoned"}">${data.prediction}</span>
    </div>
    <div class="result-features">
      ${Object.entries(data.features_used)
        .map(([key, val]) => `<span>${key}</span><span>${formatValue(val)}</span>`)
        .join("")}
    </div>
  `;
  resultBox.classList.remove("hidden");
}

function formatValue(val) {
  if (typeof val === "boolean") return val ? "yes" : "no";
  if (typeof val === "number" && !Number.isInteger(val)) return val.toFixed(2);
  return val;
}

// Dataset section: load static stats and render chart
async function loadStats() {
  const res = await fetch("data/stats.json");
  const stats = await res.json();

  document.getElementById("stat-total").textContent = stats.total_repos.toLocaleString();
  document.getElementById("stat-auc").textContent = stats.auc.toFixed(3);
  document.getElementById("stat-abandoned").textContent = `${stats.abandoned_pct}%`;

  const entries = Object.entries(stats.feature_importance).sort((a, b) => b[1] - a[1]);
  const labels = entries.map(([k]) => k);
  const values = entries.map(([, v]) => v);

  new Chart(document.getElementById("importance-chart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: "#1B7A6B",
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: "#DDE1DC" } },
        y: { grid: { display: false } },
      },
    },
  });
}

loadStats();
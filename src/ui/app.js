const output = document.getElementById("output");
const responseView = document.getElementById("response-view");
const btnHealth = document.getElementById("btn-health");
const btnModel = document.getElementById("btn-model");
const form = document.getElementById("predict-form");
const btnClear = document.getElementById("btn-clear");
const txnIdInput = document.getElementById("txn-id");
const includeFeaturesInput = document.getElementById("include-features");

async function callApi(path, options = {}) {
  const res = await fetch(`/api/v1${path}`, options);
  const text = await res.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  return { ok: res.ok, status: res.status, body };
}

function print(title, data) {
  output.textContent = `${title}\n\n${JSON.stringify(data, null, 2)}`;
  responseView.innerHTML = renderFriendly(title, data);
}

function printError(title, err) {
  const payload = {
    message: err?.message || String(err),
  };
  print(title, payload);
}

function safeText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (m) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[m]
  ));
}

function badgeForRisk(riskBand) {
  const band = String(riskBand || "low").toLowerCase();
  return `badge badge-risk-${band}`;
}

function renderCard(label, value) {
  return `
    <article class="summary-card">
      <p class="label">${safeText(label)}</p>
      <p class="value">${safeText(value)}</p>
    </article>
  `;
}

function renderHealth(data) {
  const body = data?.body || {};
  const healthy = body.status === "ok";
  return `
    <div class="summary-grid">
      ${renderCard("Service", healthy ? "Operational" : "Degraded")}
      ${renderCard("Model Version", body.model_version || "Unavailable")}
      ${renderCard("Database", body.db_connected ? "Connected" : "Disconnected")}
      ${renderCard("Uptime", `${Number(body.uptime_seconds || 0).toFixed(1)} sec`)}
    </div>
    <p><span class="${healthy ? "badge badge-ok" : "badge badge-warn"}">
      ${healthy ? "Ready for Scoring" : "Needs Attention"}
    </span></p>
  `;
}

function renderModelInfo(data) {
  const body = data?.body || {};
  const metrics = body.metrics || {};
  return `
    <div class="summary-grid">
      ${renderCard("Model", body.model_name || "N/A")}
      ${renderCard("Version", body.model_version || "N/A")}
      ${renderCard("AUC ROC", metrics.roc_auc ?? "N/A")}
      ${renderCard("F1 Score", metrics.f1 ?? "N/A")}
    </div>
    <p><span class="badge badge-ok">Model Information Loaded</span></p>
  `;
}

function renderPredict(data) {
  const body = data?.body || {};
  const score = Number(body.risk_score || 0);
  const riskBand = body?.risk_context?.risk_band || "low";
  const action = body?.risk_context?.recommended_action || "allow";
  return `
    <div class="summary-grid">
      ${renderCard("Risk Score", `${score.toFixed(2)} / 100`)}
      ${renderCard("Risk Level", riskBand.toUpperCase())}
      ${renderCard("Recommended Action", action.toUpperCase())}
      ${renderCard("Decision", body.is_fraud ? "Flagged" : "Approved")}
    </div>
    <p><span class="${badgeForRisk(riskBand)}">${safeText(riskBand.toUpperCase())} RISK</span></p>
    <div class="meter"><span style="width: ${Math.max(0, Math.min(100, score))}%;"></span></div>
  `;
}

function renderDefault() {
  return `<p class="placeholder">Response received. Expand technical details below for full payload.</p>`;
}

function renderFriendly(title, data) {
  if (title.includes("/health")) return renderHealth(data);
  if (title.includes("/model-info")) return renderModelInfo(data);
  if (title.includes("/predict")) return renderPredict(data);
  return renderDefault();
}

btnHealth.addEventListener("click", async () => {
  print("GET /api/v1/health (loading...)", {});
  try {
    const result = await callApi("/health");
    print("GET /api/v1/health", result);
  } catch (err) {
    printError("GET /api/v1/health (failed)", err);
  }
});

btnModel.addEventListener("click", async () => {
  print("GET /api/v1/model-info (loading...)", {});
  try {
    const result = await callApi("/model-info");
    print("GET /api/v1/model-info", result);
  } catch (err) {
    printError("GET /api/v1/model-info (failed)", err);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const transactionId = txnIdInput.value.trim();
  if (!transactionId) {
    print("Validation error", { message: "Transaction UUID is required." });
    return;
  }

  const payload = {
    transaction_id: transactionId,
    include_features: includeFeaturesInput.checked,
  };

  print("POST /api/v1/predict (loading...)", payload);
  try {
    const result = await callApi("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    print("POST /api/v1/predict", result);
  } catch (err) {
    printError("POST /api/v1/predict (failed)", err);
  }
});

btnClear.addEventListener("click", () => {
  output.textContent = "Run an action to see API output...";
  responseView.innerHTML = '<p class="placeholder">Run an action to see results.</p>';
});

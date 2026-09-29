"use strict";

const CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];
const CLASS_LABELS = {
  akiec: "Actinic keratoses / intraepithelial carcinoma",
  bcc: "Basal cell carcinoma",
  bkl: "Benign keratosis-like lesion",
  df: "Dermatofibroma",
  mel: "Melanoma",
  nv: "Melanocytic nevus",
  vasc: "Vascular lesion",
};

const form = document.getElementById("classify-form");
const input = document.getElementById("image-input");
const preview = document.getElementById("image-preview");
const placeholder = document.getElementById("preview-placeholder");
const fileName = document.getElementById("file-name");
const button = document.getElementById("classify-button");
const status = document.getElementById("status");
let previewUrl = null;

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function clearResult() {
  document.getElementById("result-content").hidden = true;
  document.getElementById("result-empty").hidden = false;
}

function selectedFile() {
  return input.files && input.files.length === 1 ? input.files[0] : null;
}

input.addEventListener("change", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  preview.hidden = true;
  preview.removeAttribute("src");
  placeholder.hidden = false;
  clearResult();
  setStatus("");
  const file = selectedFile();
  fileName.textContent = file ? file.name : "";
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) {
    setStatus("Please choose an image no larger than 10 MB.", true);
    return;
  }
  previewUrl = URL.createObjectURL(file);
  preview.onload = () => { preview.hidden = false; placeholder.hidden = true; };
  preview.onerror = () => { setStatus("This file cannot be previewed as an image.", true); };
  preview.src = previewUrl;
});

window.addEventListener("pagehide", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});

function validResult(data) {
  if (!data || typeof data !== "object" || Array.isArray(data)) return false;
  if (!Array.isArray(data.class_order) || data.class_order.length !== CLASS_ORDER.length ||
      !CLASS_ORDER.every((code, index) => data.class_order[index] === code)) return false;
  if (!CLASS_ORDER.includes(data.predicted_class) ||
      !Number.isInteger(data.predicted_class_index) ||
      data.class_order[data.predicted_class_index] !== data.predicted_class) return false;
  const scores = data.softmax_scores;
  if (!scores || typeof scores !== "object" || Array.isArray(scores) ||
      Object.keys(scores).length !== CLASS_ORDER.length ||
      !CLASS_ORDER.every(code => typeof scores[code] === "number" &&
        Number.isFinite(scores[code]) && scores[code] >= 0 && scores[code] <= 1)) return false;
  return typeof data.score_interpretation === "string" &&
    typeof data.intended_use === "string";
}

function addDetail(list, label, value) {
  const term = document.createElement("dt");
  term.textContent = label;
  const description = document.createElement("dd");
  description.textContent = typeof value === "string" && value ? value : "—";
  list.append(term, description);
}

function showResult(data) {
  document.getElementById("prediction").textContent = data.predicted_class;
  document.getElementById("prediction-name").textContent = CLASS_LABELS[data.predicted_class];
  const bars = document.getElementById("score-bars");
  bars.replaceChildren();
  const orderedScores = CLASS_ORDER.map(code => [code, data.softmax_scores[code]])
    .sort((a, b) => b[1] - a[1]);
  for (const [code, score] of orderedScores) {
    const row = document.createElement("div");
    row.className = "score-row";
    const heading = document.createElement("div");
    heading.className = "score-heading";
    const label = document.createElement("div");
    const abbreviation = document.createElement("strong");
    abbreviation.textContent = code;
    const name = document.createElement("span");
    name.className = "score-name";
    name.textContent = CLASS_LABELS[code];
    label.append(abbreviation, name);
    const numeric = document.createElement("span");
    numeric.textContent = `score ${score.toFixed(4)}`;
    heading.append(label, numeric);
    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = `${score * 100}%`;
    track.append(fill);
    row.append(heading, track);
    bars.append(row);
  }
  document.getElementById("score-interpretation").textContent = data.score_interpretation;
  document.getElementById("intended-use").textContent = data.intended_use;
  const provenance = data.classifier_provenance || {};
  const inputInfo = data.input || {};
  const details = document.getElementById("technical-details");
  details.replaceChildren();
  addDetail(details, "Runtime", provenance.runtime);
  addDetail(details, "Model", provenance.model_name);
  addDetail(details, "Model version", provenance.model_version);
  addDetail(details, "Preprocessing", provenance.preprocessing);
  addDetail(details, "Engine SHA-256", provenance.expected_engine_sha256);
  addDetail(details, "Input image SHA-256", inputInfo.image_sha256);
  document.getElementById("result-empty").hidden = true;
  document.getElementById("result-content").hidden = false;
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const file = selectedFile();
  if (!file) { setStatus("Choose one image before classifying.", true); return; }
  if (file.size > 10 * 1024 * 1024) {
    setStatus("Please choose an image no larger than 10 MB.", true);
    return;
  }
  button.disabled = true;
  input.disabled = true;
  button.textContent = "Classifying…";
  clearResult();
  setStatus("Classifying image…");
  try {
    const body = new FormData();
    body.append("image", file, file.name);
    const response = await fetch("/classify", { method: "POST", body });
    let data;
    try { data = await response.json(); } catch { data = null; }
    if (!response.ok) {
      const message = data && typeof data.detail === "string" ? data.detail : "Classification is unavailable right now.";
      setStatus(message, true);
    } else if (!validResult(data)) {
      setStatus("The classifier returned an unexpected response. Please try again later.", true);
    } else {
      showResult(data);
      setStatus("Classification complete. These are uncalibrated model scores.");
    }
  } catch {
    setStatus("Could not reach the classifier. Check your connection and try again.", true);
  } finally {
    button.disabled = false;
    input.disabled = false;
    button.textContent = "Classify image";
  }
});

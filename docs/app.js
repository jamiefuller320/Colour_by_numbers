const categorySelect = document.getElementById("category");
const typeSelect = document.getElementById("type");
const modelSelect = document.getElementById("model");
const sizeSelect = document.getElementById("size");
const seedInput = document.getElementById("seed");
const promptInput = document.getElementById("prompt");
const generateBtn = document.getElementById("generate");
const resetBtn = document.getElementById("reset-prompt");
const statusEl = document.getElementById("status");
const frame = document.getElementById("result-frame");
const imageEl = document.getElementById("result-image");
const openDirect = document.getElementById("open-direct");
const downloadLink = document.getElementById("download");

let categories = {};

function buildPrompt(label, category) {
  const style =
    "children's colouring book illustration, thick clean black outlines, flat cel fills, limited palette, high subject-background contrast, simple shapes, no photorealism, no text, white background";
  if (category === "aircraft") {
    return `${label} side view, clear silhouette, ${style}`;
  }
  if (category === "flowers" || category === "birds") {
    return `${label} centred portrait, ${style}`;
  }
  return `${label} portrait, centred subject, ${style}`;
}

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
}

function fillTypes() {
  const category = categorySelect.value;
  const types = categories[category] || [];
  typeSelect.innerHTML = "";
  for (const label of types) {
    const opt = document.createElement("option");
    opt.value = label;
    opt.textContent = label;
    typeSelect.appendChild(opt);
  }
  resetPrompt();
}

function resetPrompt() {
  const category = categorySelect.value;
  const label = typeSelect.value;
  promptInput.value = buildPrompt(label, category);
}

function pollinationsUrl(prompt, { width, height, model, seed }) {
  const encoded = encodeURIComponent(prompt);
  const params = new URLSearchParams({
    width: String(width),
    height: String(height),
    model,
    nologo: "true",
    enhance: "true",
  });
  if (seed !== null && seed !== undefined && seed !== "") {
    params.set("seed", String(seed));
  }
  // Cache-bust so Generate always requests a fresh image when seed is blank.
  if (seed === null || seed === undefined || seed === "") {
    params.set("_", String(Date.now()));
  }
  return `https://image.pollinations.ai/prompt/${encoded}?${params.toString()}`;
}

async function generate() {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    setStatus("Add a prompt first.", "error");
    return;
  }

  const size = Number(sizeSelect.value);
  const seedRaw = seedInput.value.trim();
  const seed = seedRaw === "" ? null : Number(seedRaw);
  const url = pollinationsUrl(prompt, {
    width: size,
    height: size,
    model: modelSelect.value,
    seed,
  });

  generateBtn.disabled = true;
  setStatus("Generating via Pollinations… this can take 15–60 seconds.");
  frame.classList.add("empty");
  imageEl.hidden = true;
  openDirect.hidden = true;
  downloadLink.hidden = true;

  try {
    // Prefer fetch so we can offer a real download blob; fall back to img src.
    const response = await fetch(url, { mode: "cors" });
    if (!response.ok) {
      throw new Error(`Pollinations returned HTTP ${response.status}`);
    }
    const blob = await response.blob();
    if (!blob.type.startsWith("image/")) {
      throw new Error(`Unexpected content type: ${blob.type || "unknown"}`);
    }
    const objectUrl = URL.createObjectURL(blob);
    imageEl.onload = () => {
      if (imageEl.dataset.prevUrl) {
        URL.revokeObjectURL(imageEl.dataset.prevUrl);
      }
      imageEl.dataset.prevUrl = objectUrl;
    };
    imageEl.src = objectUrl;
    imageEl.hidden = false;
    frame.classList.remove("empty");
    frame.querySelector(".placeholder")?.remove();

    openDirect.href = url;
    openDirect.hidden = false;
    downloadLink.href = objectUrl;
    downloadLink.download = `${typeSelect.value.replace(/\s+/g, "_") || "illustration"}.png`;
    downloadLink.hidden = false;
    setStatus(`Generated “${typeSelect.value}” (${size}×${size}, ${modelSelect.value}).`, "ok");
  } catch (err) {
    // Fallback: embed directly (works even if CORS fetch fails).
    imageEl.src = url;
    imageEl.hidden = false;
    frame.classList.remove("empty");
    frame.querySelector(".placeholder")?.remove();
    openDirect.href = url;
    openDirect.hidden = false;
    downloadLink.hidden = true;
    setStatus(
      `Showing image via direct URL. Download may be limited (${err.message}).`,
      "ok"
    );
  } finally {
    generateBtn.disabled = false;
  }
}

async function init() {
  const response = await fetch("./categories.json");
  categories = await response.json();
  const names = Object.keys(categories).sort();
  categorySelect.innerHTML = "";
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === "dogs") opt.selected = true;
    categorySelect.appendChild(opt);
  }
  fillTypes();
  categorySelect.addEventListener("change", fillTypes);
  typeSelect.addEventListener("change", resetPrompt);
  resetBtn.addEventListener("click", resetPrompt);
  generateBtn.addEventListener("click", generate);
}

init().catch((err) => {
  setStatus(`Failed to load categories: ${err.message}`, "error");
});

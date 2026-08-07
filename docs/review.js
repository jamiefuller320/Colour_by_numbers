const STORAGE_KEY = "cbn_plate_critiques_v1";

const gallery = document.getElementById("review-gallery");
const filterCategory = document.getElementById("filter-category");
const filterRating = document.getElementById("filter-rating");
const reviewerName = document.getElementById("reviewer-name");
const statusEl = document.getElementById("review-status");
const summaryEl = document.getElementById("review-summary");
const collatePreview = document.getElementById("collate-preview");
const exportBtn = document.getElementById("export-critiques");
const importInput = document.getElementById("import-critiques");
const clearBtn = document.getElementById("clear-local");

let manifest = { plates: [], issue_tags: {} };
let localCritiques = loadLocal();

function loadLocal() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveLocal() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(localCritiques));
}

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
}

function nowIso() {
  return new Date().toISOString();
}

function getCritique(plateId) {
  return localCritiques[plateId] || null;
}

function allCritiquesList() {
  return Object.values(localCritiques);
}

function collateLocal() {
  const failures = allCritiquesList().filter((c) => c.rating !== "pass");
  const byTag = {};
  const byCategory = {};
  for (const row of failures) {
    byCategory[row.category] = (byCategory[row.category] || 0) + 1;
    for (const tag of row.issues || []) {
      byTag[tag] = (byTag[tag] || 0) + 1;
    }
  }
  return { failures, byTag, byCategory, total: allCritiquesList().length };
}

function updateSummary() {
  const { failures, byTag, byCategory, total } = collateLocal();
  const reviewed = total;
  const pending = manifest.plates.length - reviewed;
  summaryEl.textContent =
    `${manifest.plates.length} plates · ${reviewed} reviewed · ` +
    `${failures.length} need improvement · ${Math.max(0, pending)} pending`;
  if (!total) {
    collatePreview.textContent = "No critiques yet.";
    return;
  }
  const lines = [
    `Reviewed: ${total}`,
    `Failures: ${failures.length}`,
    `By category: ${JSON.stringify(byCategory)}`,
    `By issue tag: ${JSON.stringify(byTag)}`,
    "",
    "Recent notes:",
  ];
  for (const row of failures.slice(-8)) {
    const note = (row.notes || row.suggested_prompt || "").slice(0, 100);
    lines.push(`  [${row.category}/${row.subject}] ${row.issues?.join(", ") || "—"} — ${note}`);
  }
  collatePreview.textContent = lines.join("\n");
}

function filteredPlates() {
  const cat = filterCategory.value;
  const rating = filterRating.value;
  return manifest.plates.filter((plate) => {
    if (cat && plate.category !== cat) return false;
    const critique = getCritique(plate.id);
    if (!rating) return true;
    if (rating === "unreviewed") return !critique;
    return critique && critique.rating === rating;
  });
}

function renderGallery() {
  const plates = filteredPlates();
  gallery.innerHTML = "";
  if (!plates.length) {
    gallery.innerHTML = '<p class="fineprint">No plates match the current filter.</p>';
    return;
  }

  for (const plate of plates) {
    const critique = getCritique(plate.id) || {};
    const card = document.createElement("article");
    card.className = "set-card review-card";
    card.dataset.plateId = plate.id;

    const title = document.createElement("h3");
    title.textContent = `${plate.label} (${plate.category})`;

    const meta = document.createElement("p");
    meta.className = "meta";
    meta.textContent = `${plate.id} · ${plate.backend || "unknown"} backend`;

    const pair = document.createElement("div");
    pair.className = "pair";
    for (const [label, key] of [
      ["Flat plate", "plate"],
      ["Outline", "outline"],
    ]) {
      const figure = document.createElement("figure");
      const img = document.createElement("img");
      img.src = `review/${plate.images[key]}`;
      img.alt = `${plate.label} ${label}`;
      img.loading = "lazy";
      const cap = document.createElement("figcaption");
      cap.textContent = label;
      figure.append(img, cap);
      pair.appendChild(figure);
    }

    const form = document.createElement("div");
    form.className = "review-form";

    const ratingLabel = document.createElement("label");
    ratingLabel.textContent = "Rating";
    const ratingSelect = document.createElement("select");
    ratingSelect.className = "review-rating";
    for (const [value, text] of [
      ["", "— not reviewed —"],
      ["pass", "Pass"],
      ["needs_work", "Needs work"],
      ["fail", "Fail"],
    ]) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = text;
      if (critique.rating === value) opt.selected = true;
      ratingSelect.appendChild(opt);
    }
    ratingLabel.append(ratingSelect);

    const issuesField = document.createElement("fieldset");
    issuesField.className = "review-issues";
    const legend = document.createElement("legend");
    legend.textContent = "Issues";
    issuesField.appendChild(legend);
    const issueGrid = document.createElement("div");
    issueGrid.className = "issue-grid";
    const selected = new Set(critique.issues || []);
    for (const [tag, description] of Object.entries(manifest.issue_tags || {})) {
      const label = document.createElement("label");
      label.className = "check issue-check";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = tag;
      input.checked = selected.has(tag);
      label.append(input, document.createTextNode(description));
      issueGrid.appendChild(label);
    }
    issuesField.appendChild(issueGrid);

    const notesLabel = document.createElement("label");
    notesLabel.className = "full";
    notesLabel.textContent = "What is wrong or missing?";
    const notesArea = document.createElement("textarea");
    notesArea.rows = 3;
    notesArea.className = "review-notes";
    notesArea.placeholder = "e.g. No nose wrinkles or nostril definition on the pug muzzle";
    notesArea.value = critique.notes || "";
    notesLabel.append(notesArea);

    const promptLabel = document.createElement("label");
    promptLabel.className = "full";
    promptLabel.textContent = "Suggested prompt addition";
    const promptArea = document.createElement("textarea");
    promptArea.rows = 2;
    promptArea.className = "review-prompt";
    promptArea.placeholder = "e.g. clearly defined nose with visible nostrils and muzzle wrinkles";
    promptArea.value = critique.suggested_prompt || "";
    promptLabel.append(promptArea);

    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.textContent = "Save critique";
    saveBtn.addEventListener("click", () => {
      const rating = ratingSelect.value;
      if (!rating) {
        delete localCritiques[plate.id];
        saveLocal();
        setStatus(`Cleared review for ${plate.label}.`);
        updateSummary();
        return;
      }
      const issues = [...issueGrid.querySelectorAll("input:checked")].map((el) => el.value);
      localCritiques[plate.id] = {
        plate_id: plate.id,
        category: plate.category,
        subject: plate.subject,
        rating,
        issues,
        notes: notesArea.value.trim(),
        suggested_prompt: promptArea.value.trim(),
        reviewer: reviewerName.value.trim(),
        reviewed_at: nowIso(),
        prompt_used: plate.prompt || "",
      };
      saveLocal();
      setStatus(`Saved critique for ${plate.label}.`, "ok");
      updateSummary();
    });

    form.append(ratingLabel, issuesField, notesLabel, promptLabel, saveBtn);
    card.append(title, meta, pair, form);
    gallery.appendChild(card);
  }
}

function populateFilters() {
  const cats = [...new Set(manifest.plates.map((p) => p.category))].sort();
  filterCategory.innerHTML = '<option value="">All categories</option>';
  for (const cat of cats) {
    const opt = document.createElement("option");
    opt.value = cat;
    opt.textContent = cat;
    filterCategory.appendChild(opt);
  }
}

exportBtn.addEventListener("click", () => {
  const payload = {
    exported_at: nowIso(),
    critiques: allCritiquesList(),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `plate-critiques-${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  URL.revokeObjectURL(url);
  setStatus(`Exported ${payload.critiques.length} critique(s).`, "ok");
});

importInput.addEventListener("change", async () => {
  const file = importInput.files?.[0];
  if (!file) return;
  try {
    const payload = await file.text();
    const data = JSON.parse(payload);
    const rows = Array.isArray(data) ? data : data.critiques || [];
    let merged = 0;
    for (const row of rows) {
      if (!row.plate_id) continue;
      localCritiques[row.plate_id] = row;
      merged += 1;
    }
    saveLocal();
    renderGallery();
    updateSummary();
    setStatus(`Imported ${merged} critique(s) into this browser.`, "ok");
  } catch (err) {
    setStatus(err.message || String(err), "error");
  } finally {
    importInput.value = "";
  }
});

clearBtn.addEventListener("click", () => {
  if (!confirm("Clear all local plate critiques?")) return;
  localCritiques = {};
  saveLocal();
  renderGallery();
  updateSummary();
  setStatus("Local critiques cleared.");
});

filterCategory.addEventListener("change", renderGallery);
filterRating.addEventListener("change", renderGallery);

async function init() {
  const response = await fetch("./review/manifest.json");
  if (!response.ok) throw new Error(`manifest.json HTTP ${response.status}`);
  manifest = await response.json();
  populateFilters();
  renderGallery();
  updateSummary();
  setStatus(`${manifest.plates.length} sample plate(s) loaded. Add critiques and export when done.`);
}

init().catch((err) => setStatus(`Failed to load review gallery: ${err.message}`, "error"));

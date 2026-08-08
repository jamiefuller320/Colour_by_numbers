const statusEl = document.getElementById("library-status");
const gridEl = document.getElementById("library-grid");
const galleryEl = document.getElementById("library-gallery");
const pairsEl = document.getElementById("gallery-pairs");
const titleEl = document.getElementById("gallery-title");
const metaEl = document.getElementById("gallery-meta");
const noteEl = document.getElementById("gallery-note");
const backBtn = document.getElementById("library-back");

let sets = [];

function setStatus(message, kind = "") {
  statusEl.textContent = message || "";
  statusEl.className = `status ${kind}`.trim();
}

function swatchHtml(hexes) {
  if (!hexes || !hexes.length) return "";
  return `<div class="library-swatches" aria-hidden="true">${hexes
    .map(
      (h) =>
        `<span class="library-swatch" style="background:${h}" title="${h}"></span>`
    )
    .join("")}</div>`;
}

function parseHash() {
  const raw = (location.hash || "").replace(/^#/, "");
  if (!raw) return { view: "grid", setId: null };
  const match = raw.match(/^set\/(.+)$/);
  if (match) return { view: "gallery", setId: decodeURIComponent(match[1]) };
  return { view: "grid", setId: null };
}

function findSet(setId) {
  return sets.find((s) => s.set_id === setId) || null;
}

function renderGrid() {
  galleryEl.hidden = true;
  gridEl.hidden = false;
  gridEl.innerHTML = "";
  if (!sets.length) {
    setStatus("No published sets in library.json yet.", "error");
    return;
  }
  setStatus(`${sets.length} set${sets.length === 1 ? "" : "s"}`);
  for (const row of sets) {
    const card = document.createElement("article");
    card.className = "library-tile";
    const cats = (row.categories || []).join(", ") || "—";
    const style = row.style || "—";
    const thumb = row.thumbnail
      ? `<img src="${row.thumbnail}" alt="${row.title} colour plate" loading="lazy" />`
      : `<div class="library-tile-empty">No preview</div>`;
    card.innerHTML = `
      <a class="library-tile-link" href="#set/${encodeURIComponent(row.set_id)}">
        <div class="library-tile-thumb">${thumb}</div>
        ${swatchHtml(row.thumbnail_colours)}
        <h2>${row.title}</h2>
        <p class="meta">${row.n_pairs} plate${row.n_pairs === 1 ? "" : "s"} · ${style} · ${cats}</p>
      </a>
    `;
    gridEl.appendChild(card);
  }
}

function assetFigure(label, src) {
  if (!src) return "";
  return `
    <figure class="library-shot">
      <h3>${label}</h3>
      <a href="${src}" target="_blank" rel="noopener">
        <img src="${src}" alt="${label}" loading="lazy" />
      </a>
      <figcaption><a href="${src}" target="_blank" rel="noopener">Open image</a></figcaption>
    </figure>
  `;
}

function renderGallery(setId) {
  const row = findSet(setId);
  if (!row) {
    setStatus(`Unknown set “${setId}”.`, "error");
    location.hash = "";
    return;
  }
  gridEl.hidden = true;
  galleryEl.hidden = false;
  setStatus("");
  titleEl.textContent = row.title;
  const bits = [
    `\`${row.set_id}\``,
    row.mode || "single",
    row.style || "no style",
    `${row.n_pairs} pair${row.n_pairs === 1 ? "" : "s"}`,
  ];
  if (row.categories && row.categories.length) {
    bits.push(row.categories.join(", "));
  }
  metaEl.textContent = bits.join(" · ");
  noteEl.textContent = row.note || "";
  noteEl.hidden = !row.note;

  pairsEl.innerHTML = "";
  for (const pair of row.pairs || []) {
    const section = document.createElement("article");
    section.className = "library-pair";
    const subject = pair.subject || "plate";
    const assets = pair.assets || {};
    section.innerHTML = `
      <h3>#${String(pair.index).padStart(2, "0")} — ${subject}</h3>
      <div class="library-pair-grid">
        ${assetFigure("Colour plate", assets.plate)}
        ${assetFigure("Numbered outline", assets.outline)}
        ${assetFigure("Print page", assets.page)}
        ${assetFigure("Source illustration", assets.illustration)}
      </div>
    `;
    pairsEl.appendChild(section);
  }
}

function syncView() {
  const { view, setId } = parseHash();
  if (view === "gallery" && setId) {
    renderGallery(setId);
  } else {
    renderGrid();
  }
}

backBtn.addEventListener("click", () => {
  location.hash = "";
});

window.addEventListener("hashchange", syncView);

async function init() {
  setStatus("Loading library…");
  try {
    const res = await fetch("library.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    sets = Array.isArray(data.sets) ? data.sets : [];
    syncView();
  } catch (err) {
    setStatus(`Failed to load library.json: ${err.message}`, "error");
  }
}

init();

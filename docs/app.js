const categorySelect = document.getElementById("category");
const typeSelect = document.getElementById("type");
const modelSelect = document.getElementById("model");
const sizeSelect = document.getElementById("size");
const seedInput = document.getElementById("seed");
const coloursInput = document.getElementById("colours");
const promptInput = document.getElementById("prompt");
const generateBtn = document.getElementById("generate");
const resetBtn = document.getElementById("reset-prompt");
const statusEl = document.getElementById("status");
const frame = document.getElementById("result-frame");
const imageEl = document.getElementById("result-image");
const outlineFrame = document.getElementById("outline-frame");
const outlineImageEl = document.getElementById("outline-image");
const openDirect = document.getElementById("open-direct");
const downloadLink = document.getElementById("download");
const downloadOutline = document.getElementById("download-outline");

// Standard crayon set aligned with Python STANDARD_PALETTE_32 (truncated to 24
// for the browser path; darks / earths are kept dense for animal subjects).
const STANDARD_PALETTE = [
  [18, 18, 18],
  [42, 36, 32],
  [78, 72, 68],
  [150, 148, 145],
  [240, 238, 232],
  [255, 230, 80],
  [245, 180, 40],
  [230, 120, 30],
  [220, 40, 40],
  [240, 120, 160],
  [55, 30, 16],
  [90, 50, 30],
  [130, 85, 45],
  [175, 130, 75],
  [220, 185, 135],
  [50, 150, 60],
  [180, 220, 120],
  [40, 180, 190],
  [50, 110, 210],
  [130, 180, 240],
  [110, 70, 160],
  [190, 155, 220],
  [255, 100, 50],
  [0, 150, 110],
];

const EARTHY_CATEGORIES = new Set([
  "dogs",
  "cats",
  "horses",
  "birds",
  "wildlife",
  "animals",
  "pets",
  "farm animals",
  "mammals",
]);
const MIN_COLOURS = 8;
const MAX_COLOURS = 16;
const MIN_REGION_MM = 5;
const A4_MM = [210, 297];
const OUTLINE_LINE_WIDTH = 1; // single-pixel edges (no thicken pass)
const DETAIL_INK_RGB = [18, 18, 18];

let categories = {};

function clampColours(n) {
  const value = Number(n);
  if (!Number.isFinite(value)) return 12;
  return Math.max(MIN_COLOURS, Math.min(MAX_COLOURS, Math.round(value)));
}

function buildPrompt(label, category, nColours = 12) {
  const colours = clampColours(nColours);
  const style =
    `children's colouring book illustration, thick clean black outlines, ` +
    `flat cel fills using between ${MIN_COLOURS} and ${colours} solid colours only, ` +
    `colourable blocks at least ${MIN_REGION_MM}mm wide and ${MIN_REGION_MM}mm high ` +
    `when printed on A4 with finer detail as black line drawing, ` +
    `high subject-background contrast, no gradients, no photorealism, no text, white background`;
  if (category === "aircraft") {
    return `${label} side view, clear silhouette, ${style}`;
  }
  if (category === "flowers") {
    return `${label} centred portrait, ${style}`;
  }
  if (EARTHY_CATEGORIES.has(category)) {
    const detail =
      "clearly defined eyes with dark pupils and light eye highlights, warm natural colours";
    if (category === "birds") {
      return `${label} centred portrait, ${detail}, ${style}`;
    }
    return `${label} portrait, centred subject, ${detail}, ${style}`;
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
  promptInput.value = buildPrompt(label, category, coloursInput?.value);
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

function mmPerPixelOnA4(width, height) {
  const [shortMm, longMm] = A4_MM;
  const scalePortrait = Math.min(shortMm / width, longMm / height);
  const scaleLandscape = Math.min(longMm / width, shortMm / height);
  return Math.max(scalePortrait, scaleLandscape);
}

function minRegionSidePx(width, height, minMm = MIN_REGION_MM) {
  const mmPerPx = mmPerPixelOnA4(width, height);
  return Math.max(1, Math.ceil(minMm / mmPerPx));
}

function colourDistance2(a, b) {
  const dr = a[0] - b[0];
  const dg = a[1] - b[1];
  const db = a[2] - b[2];
  return dr * dr + dg * dg + db * db;
}

function relativeLuminance(rgb) {
  // Cheap luma; good enough to gate low-light fur remapping.
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

function isEarthyShadowColour(rgb) {
  const [r, g, b] = rgb;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const chroma = max - min;
  const luma = relativeLuminance(rgb);
  if (chroma <= 28) return true;
  // Warm browns: red channel leads, not too vivid.
  if (luma < 150 && r >= g && g >= b - 8 && chroma <= 110) return true;
  return false;
}

function selectActivePalette(imageData, nColours, category = null) {
  const n = clampColours(nColours);
  const counts = new Array(STANDARD_PALETTE.length).fill(0);
  const { data } = imageData;
  for (let i = 0; i < data.length; i += 16) {
    // Sample every 4th pixel for speed.
    const rgb = [data[i], data[i + 1], data[i + 2]];
    let best = 0;
    let bestDist = Infinity;
    for (let p = 0; p < STANDARD_PALETTE.length; p += 1) {
      const dist = colourDistance2(rgb, STANDARD_PALETTE[p]);
      if (dist < bestDist) {
        bestDist = dist;
        best = p;
      }
    }
    counts[best] += 1;
  }

  if (EARTHY_CATEGORIES.has(category)) {
    for (let p = 0; p < STANDARD_PALETTE.length; p += 1) {
      if (!isEarthyShadowColour(STANDARD_PALETTE[p])) {
        counts[p] *= 0.2;
      }
    }
  }

  const order = counts
    .map((count, idx) => ({ count, idx }))
    .sort((a, b) => b.count - a.count);

  const chosenIdx = [];
  if (EARTHY_CATEGORIES.has(category)) {
    // Reserve warm darks + a light paper tone.
    for (const idx of [0, 1, 2, 10, 11, 4]) {
      if (chosenIdx.length >= Math.min(6, n)) break;
      if (!chosenIdx.includes(idx)) chosenIdx.push(idx);
    }
  }
  for (const item of order) {
    if (chosenIdx.length >= n) break;
    if (!chosenIdx.includes(item.idx)) chosenIdx.push(item.idx);
  }
  while (chosenIdx.length < Math.max(MIN_COLOURS, Math.min(n, MAX_COLOURS))) {
    for (let i = 0; i < STANDARD_PALETTE.length; i += 1) {
      if (!chosenIdx.includes(i)) {
        chosenIdx.push(i);
        break;
      }
    }
  }
  return chosenIdx
    .slice(0, Math.max(MIN_COLOURS, Math.min(n, MAX_COLOURS)))
    .map((idx) => STANDARD_PALETTE[idx]);
}

function nearestPaletteIndex(rgb, palette, category = null) {
  const dark = EARTHY_CATEGORIES.has(category) && relativeLuminance(rgb) < 95;
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < palette.length; i += 1) {
    if (dark && !isEarthyShadowColour(palette[i])) continue;
    const dist = colourDistance2(rgb, palette[i]);
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  // Fallback if every candidate was filtered (should be rare).
  if (bestDist === Infinity) {
    for (let i = 0; i < palette.length; i += 1) {
      const dist = colourDistance2(rgb, palette[i]);
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
  }
  return best;
}

function neighboursOf(idx, width, height) {
  const x = idx % width;
  const y = (idx / width) | 0;
  const out = [];
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      if (dx === 0 && dy === 0) continue;
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
      out.push(ny * width + nx);
    }
  }
  return out;
}

function absorbSmallLabels(labels, width, height, minArea, maxPasses = 8) {
  const size = width * height;
  let work = labels.slice();

  for (let pass = 0; pass < maxPasses; pass += 1) {
    const visited = new Uint8Array(size);
    let absorbed = 0;
    const next = work.slice();

    for (let start = 0; start < size; start += 1) {
      if (visited[start]) continue;
      const colour = work[start];
      const stack = [start];
      const component = [];
      visited[start] = 1;
      while (stack.length) {
        const idx = stack.pop();
        component.push(idx);
        for (const n of neighboursOf(idx, width, height)) {
          if (visited[n] || work[n] !== colour) continue;
          visited[n] = 1;
          stack.push(n);
        }
      }
      if (component.length >= minArea) continue;

      const votes = new Map();
      for (const idx of component) {
        for (const n of neighboursOf(idx, width, height)) {
          const other = work[n];
          if (other === colour) continue;
          votes.set(other, (votes.get(other) || 0) + 1);
        }
      }
      let bestColour = colour;
      let bestVotes = 0;
      for (const [c, v] of votes.entries()) {
        if (v > bestVotes) {
          bestVotes = v;
          bestColour = c;
        }
      }
      if (bestVotes === 0) continue;
      for (const idx of component) next[idx] = bestColour;
      absorbed += 1;
    }
    work = next;
    if (absorbed === 0) break;
  }
  return work;
}

function componentBBox(component, width) {
  let minX = width;
  let maxX = 0;
  let minY = 1e9;
  let maxY = 0;
  for (const idx of component) {
    const x = idx % width;
    const y = (idx / width) | 0;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return { w: maxX - minX + 1, h: maxY - minY + 1 };
}

function approxInscribedDiameter(component, width, height, labels, colour) {
  // Cheaper than a full EDT: max Chebyshev distance to a border-ish neighbour.
  let best = 0;
  for (const idx of component) {
    const x = idx % width;
    const y = (idx / width) | 0;
    let onBorder = false;
    for (const n of neighboursOf(idx, width, height)) {
      if (labels[n] !== colour) {
        onBorder = true;
        break;
      }
    }
    if (onBorder) continue;
    const depth = Math.min(x + 1, y + 1, width - x, height - y);
    if (depth > best) best = depth;
  }
  // Interior depth ≈ radius; diameter ≈ 2 * depth. Border-only blobs → 1px.
  return best > 0 ? best * 2 : 1;
}

function detailInkFromComponent(component, width, height, labels, colour) {
  const diameter = approxInscribedDiameter(component, width, height, labels, colour);
  if (diameter <= 2.5) return component.slice();
  const edge = [];
  for (const idx of component) {
    let border = false;
    for (const n of neighboursOf(idx, width, height)) {
      if (labels[n] !== colour) {
        border = true;
        break;
      }
    }
    // Also treat image-edge pixels as outline candidates.
    const x = idx % width;
    const y = (idx / width) | 0;
    if (x === 0 || y === 0 || x === width - 1 || y === height - 1) border = true;
    if (border) edge.push(idx);
  }
  return edge.length ? edge : component.slice();
}

function enforceColourableBlocks(
  labels,
  width,
  height,
  minWidthPx,
  minHeightPx,
  maxPasses = 10
) {
  const size = width * height;
  let work = labels.slice();
  const detail = new Uint8Array(size);
  const minInscribed = Math.min(minWidthPx, minHeightPx);

  for (let pass = 0; pass < maxPasses; pass += 1) {
    const visited = new Uint8Array(size);
    let changed = 0;
    const next = work.slice();

    for (let start = 0; start < size; start += 1) {
      if (visited[start]) continue;
      const colour = work[start];
      const stack = [start];
      const component = [];
      visited[start] = 1;
      while (stack.length) {
        const idx = stack.pop();
        component.push(idx);
        for (const n of neighboursOf(idx, width, height)) {
          if (visited[n] || work[n] !== colour) continue;
          visited[n] = 1;
          stack.push(n);
        }
      }
      const { w, h } = componentBBox(component, width);
      const inscribed = approxInscribedDiameter(
        component,
        width,
        height,
        work,
        colour
      );
      if (w >= minWidthPx && h >= minHeightPx && inscribed >= minInscribed) {
        continue;
      }

      for (const idx of detailInkFromComponent(
        component,
        width,
        height,
        work,
        colour
      )) {
        detail[idx] = 1;
      }

      const votes = new Map();
      for (const idx of component) {
        for (const n of neighboursOf(idx, width, height)) {
          const other = work[n];
          if (other === colour) continue;
          votes.set(other, (votes.get(other) || 0) + 1);
        }
      }
      let bestColour = colour;
      let bestVotes = 0;
      for (const [c, v] of votes.entries()) {
        if (v > bestVotes) {
          bestVotes = v;
          bestColour = c;
        }
      }
      if (bestVotes === 0) continue;
      for (const idx of component) next[idx] = bestColour;
      changed += 1;
    }
    work = next;
    if (changed === 0) break;
  }
  return { labels: work, detail };
}

function compactLabels(labels, palette) {
  const used = [];
  const seen = new Set();
  for (let i = 0; i < labels.length; i += 1) {
    const value = labels[i];
    if (!seen.has(value)) {
      seen.add(value);
      used.push(value);
    }
  }
  used.sort((a, b) => a - b);
  const remap = new Map(used.map((old, idx) => [old, idx]));
  const compacted = new Int16Array(labels.length);
  for (let i = 0; i < labels.length; i += 1) {
    compacted[i] = remap.get(labels[i]);
  }
  return {
    labels: compacted,
    palette: used.map((idx) => palette[idx]),
  };
}

function listRegions(labels, width, height) {
  const size = width * height;
  const visited = new Uint8Array(size);
  const regions = [];

  for (let start = 0; start < size; start += 1) {
    if (visited[start]) continue;
    const colour = labels[start];
    const stack = [start];
    const component = [];
    visited[start] = 1;
    let sumX = 0;
    let sumY = 0;
    while (stack.length) {
      const idx = stack.pop();
      component.push(idx);
      sumX += idx % width;
      sumY += (idx / width) | 0;
      for (const n of neighboursOf(idx, width, height)) {
        if (visited[n] || labels[n] !== colour) continue;
        visited[n] = 1;
        stack.push(n);
      }
    }

    // Prefer an interior anchor near the component centre.
    let bestIdx = component[0];
    let bestDist = -Infinity;
    const cx = sumX / component.length;
    const cy = sumY / component.length;
    for (const idx of component) {
      const x = idx % width;
      const y = (idx / width) | 0;
      let onBorder = false;
      for (const n of neighboursOf(idx, width, height)) {
        if (labels[n] !== colour) {
          onBorder = true;
          break;
        }
      }
      const depth = Math.min(x, y, width - 1 - x, height - 1 - y);
      const toCentre = -((x - cx) * (x - cx) + (y - cy) * (y - cy));
      const score = (onBorder ? -1000 : 0) + depth * 10 + toCentre;
      if (score > bestDist) {
        bestDist = score;
        bestIdx = idx;
      }
    }

    regions.push({
      colourIndex: colour,
      number: colour + 1,
      area: component.length,
      x: bestIdx % width,
      y: (bestIdx / width) | 0,
    });
  }
  return regions;
}

function prepareIllustrationCanvas(sourceImage, nColours, category = null) {
  const width = sourceImage.naturalWidth || sourceImage.width;
  const height = sourceImage.naturalHeight || sourceImage.height;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(sourceImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, width, height);
  const palette = selectActivePalette(imageData, nColours, category);
  const labels = new Int16Array(width * height);
  const { data } = imageData;
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    labels[p] = nearestPaletteIndex(
      [data[i], data[i + 1], data[i + 2]],
      palette,
      category
    );
  }
  const side = minRegionSidePx(width, height, MIN_REGION_MM);
  const cleaned = absorbSmallLabels(labels, width, height, side * side);
  const enforced = enforceColourableBlocks(cleaned, width, height, side, side);
  const compacted = compactLabels(enforced.labels, palette);
  // Remap detail through compaction: rebuild from compacted labels by
  // re-running enforce is expensive; instead paint detail on the compacted
  // grid using the pre-compact pixel mask (indices unchanged by remapping
  // of colours — detail is positional).
  for (let p = 0; p < compacted.labels.length; p += 1) {
    const colour = enforced.detail[p]
      ? DETAIL_INK_RGB
      : compacted.palette[compacted.labels[p]];
    const o = p * 4;
    data[o] = colour[0];
    data[o + 1] = colour[1];
    data[o + 2] = colour[2];
    data[o + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return {
    canvas,
    labels: compacted.labels,
    palette: compacted.palette,
    detail: enforced.detail,
    usedColours: compacted.palette.length,
    minSidePx: side,
    width,
    height,
  };
}

function buildOutlineCanvas(labels, palette, width, height, detail = null) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height);
  const { data } = imageData;
  const paintEdge = (idx) => {
    const o = idx * 4;
    data[o] = 0;
    data[o + 1] = 0;
    data[o + 2] = 0;
    data[o + 3] = 255;
  };

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      const colour = labels[idx];
      if (x > 0 && labels[idx - 1] !== colour) paintEdge(idx);
      if (y > 0 && labels[idx - width] !== colour) paintEdge(idx);
      if (detail && detail[idx]) paintEdge(idx);
    }
  }
  // Optional thicker outline only when explicitly requested.
  if (OUTLINE_LINE_WIDTH > 1) {
    const edgeMask = new Uint8Array(width * height);
    for (let i = 0; i < edgeMask.length; i += 1) {
      if (data[i * 4] === 0 && data[i * 4 + 1] === 0 && data[i * 4 + 2] === 0) {
        edgeMask[i] = 1;
      }
    }
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const idx = y * width + x;
        if (!edgeMask[idx]) continue;
        for (const n of [idx - 1, idx + 1, idx - width, idx + width]) {
          paintEdge(n);
        }
      }
    }
  }
  ctx.putImageData(imageData, 0, 0);

  const regions = listRegions(labels, width, height);
  const baseFont = Math.max(12, Math.round(Math.min(width, height) * 0.035));
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // Draw larger regions first so small-block numbers stay visible on top.
  regions.sort((a, b) => b.area - a.area);
  for (const region of regions) {
    const side = Math.sqrt(region.area);
    const fontSize = Math.max(7, Math.min(baseFont, Math.round(side * 0.45)));
    const text = String(region.number);
    ctx.font = `bold ${fontSize}px "Source Sans 3", "Segoe UI", sans-serif`;
    ctx.lineWidth = Math.max(2, Math.round(fontSize / 5));
    ctx.strokeStyle = "#ffffff";
    ctx.fillStyle = "#000000";
    ctx.strokeText(text, region.x, region.y);
    ctx.fillText(text, region.x, region.y);
  }

  return {
    canvas,
    regionCount: regions.length,
    colourCount: palette.length,
  };
}

function blobFromCanvas(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) reject(new Error("Could not encode PNG"));
      else resolve(blob);
    }, "image/png");
  });
}

function revokePrev(el) {
  if (el?.dataset?.prevUrl) {
    URL.revokeObjectURL(el.dataset.prevUrl);
    delete el.dataset.prevUrl;
  }
}

function showImage(el, frameEl, objectUrl) {
  revokePrev(el);
  el.onload = () => {
    el.dataset.prevUrl = objectUrl;
  };
  el.src = objectUrl;
  el.hidden = false;
  frameEl.classList.remove("empty");
  frameEl.querySelector(".placeholder")?.remove();
}

function clearOutlinePreview() {
  outlineImageEl.hidden = true;
  outlineFrame.classList.add("empty");
  if (!outlineFrame.querySelector(".placeholder")) {
    const p = document.createElement("p");
    p.className = "placeholder outline-placeholder";
    p.textContent = "The numbered outline appears here after generation.";
    outlineFrame.appendChild(p);
  }
  downloadOutline.hidden = true;
  revokePrev(outlineImageEl);
  outlineImageEl.removeAttribute("src");
}

async function generate() {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    setStatus("Add a prompt first.", "error");
    return;
  }

  const size = Number(sizeSelect.value);
  const nColours = clampColours(coloursInput?.value || 12);
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
  clearOutlinePreview();

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

    const rawUrl = URL.createObjectURL(blob);
    const rawImage = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Failed to decode generated image"));
      img.src = rawUrl;
    });

    setStatus("Clamping to 8–16 flat colours; colourable blocks ≥5×5mm…");
    const prepared = prepareIllustrationCanvas(
      rawImage,
      nColours,
      categorySelect.value
    );
    const preparedBlob = await blobFromCanvas(prepared.canvas);
    URL.revokeObjectURL(rawUrl);
    const objectUrl = URL.createObjectURL(preparedBlob);
    showImage(imageEl, frame, objectUrl);

    setStatus("Building numbered outline with line detail…");
    const outline = buildOutlineCanvas(
      prepared.labels,
      prepared.palette,
      prepared.width,
      prepared.height,
      prepared.detail
    );
    const outlineBlob = await blobFromCanvas(outline.canvas);
    const outlineUrl = URL.createObjectURL(outlineBlob);
    showImage(outlineImageEl, outlineFrame, outlineUrl);

    openDirect.href = url;
    openDirect.hidden = false;
    const stem = typeSelect.value.replace(/\s+/g, "_") || "illustration";
    downloadLink.href = objectUrl;
    downloadLink.download = `${stem}.png`;
    downloadLink.hidden = false;
    downloadOutline.href = outlineUrl;
    downloadOutline.download = `${stem}_outline.png`;
    downloadOutline.hidden = false;
    setStatus(
      `Generated “${typeSelect.value}” · ${prepared.usedColours} colours · ` +
        `${outline.regionCount} colourable blocks · ` +
        `min block ${prepared.minSidePx}×${prepared.minSidePx}px ` +
        `(≥${MIN_REGION_MM}mm wide & high on A4).`,
      "ok"
    );
  } catch (err) {
    // Fallback: embed directly (works even if CORS fetch fails).
    imageEl.src = url;
    imageEl.hidden = false;
    frame.classList.remove("empty");
    frame.querySelector(".placeholder")?.remove();
    openDirect.href = url;
    openDirect.hidden = false;
    downloadLink.hidden = true;
    clearOutlinePreview();
    setStatus(
      `Showing raw Pollinations image (post-process skipped: ${err.message}).`,
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
  coloursInput?.addEventListener("change", resetPrompt);
  resetBtn.addEventListener("click", resetPrompt);
  generateBtn.addEventListener("click", generate);
}

init().catch((err) => {
  setStatus(`Failed to load categories: ${err.message}`, "error");
});

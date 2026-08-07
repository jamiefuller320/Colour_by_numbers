# Static plate viewer (GitHub Pages)

This folder is published to **GitHub Pages**. It is a **viewer only**: browse
example plates from the live fal.ai pipeline, or upload your own generated
colour plate, clamp it to the colouring palette, and preview the numbered
outline. Works on **iPad / phone** with no localhost and **no API keys**.

**Generation** (fal.ai Flux) runs in Streamlit or the CLI with `FAL_KEY` — not
in the browser.

Example assets live under `samples/` and are listed in `samples.json`.

## Local preview

```bash
cd docs
python3 -m http.server 8080
# open http://localhost:8080
```

## Enable Pages on the repo

1. Repo **Settings → Pages**
2. **Source**: GitHub Actions
3. Merge to `main` (or run the “Deploy GitHub Pages test bed” workflow)

Site URL: `https://jamiefuller320.github.io/Colour_by_numbers/`

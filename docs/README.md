# Static illustration test bed

This folder is published to **GitHub Pages**. It calls [Pollinations.ai](https://pollinations.ai) from the browser — no Python server. Works on **iPad / phone** (no localhost).

**Pollinations auth:** anonymous generation currently fails (HTTP 500 wrapping “insufficient Pollen”). Create a key and add a little credit at [enter.pollinations.ai](https://enter.pollinations.ai), paste the key into the page (stored in localStorage only).

Supports single plates and **Phase D set mode** (varied aspect/scene plates, sequential generation with a ~16s gap).

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

On the live site: enable **Set mode**, pick set size, generate — plates appear in a scrollable gallery with downloads.

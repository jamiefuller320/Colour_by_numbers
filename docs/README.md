# Static illustration test bed

This folder is published to **GitHub Pages**. It calls [Pollinations.ai](https://pollinations.ai) from the browser — no Python server and no paid API subscription. Works on **iPad / phone** (no localhost).

Supports single plates and **Phase D set mode** (varied aspect/scene plates, sequential generation with a ~16s gap for Pollinations rate limits).

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

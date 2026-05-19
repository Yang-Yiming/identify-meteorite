# Encyclopedia of Meteorites crawler

Polite Playwright crawler for public pages on `encyclopedia-of-meteorites.com`.

## Setup

```bash
npm install
```

## Run

Small default run:

```bash
npm run crawl:eom
```

Visible browser:

```bash
npm run crawl:eom:headed
```

Useful options:

```bash
node crawl-eom.mjs --max-pages 5 --max-items 50 --delay 2000
node crawl-eom.mjs --out output/eom-small --max-items 20
node crawl-eom.mjs --channel chrome --headed
```

Output:

- `output/eom/images/`: downloaded image files
- `output/eom/pages/<id>/page.json`: per-meteorite metadata
- `output/eom/manifest.json`: run manifest
- `output/eom/state.json`: resume state

The crawler is intentionally conservative: it only follows public links, runs one browser page at a time, delays between visits, and resumes from saved state.

By default it uses the installed Microsoft Edge channel (`msedge`) so Playwright does not need to download its own Chromium build. Use `--channel chrome` if Chrome is your preferred installed browser.

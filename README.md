# Newsletter Builder

A Flask web app that builds email-safe HTML newsletter blocks for House of Hamill and Enter the Haggis: tour dates from Bandsintown, merch scraped from the band site, and a generated US tour map.

Deployed on Render at https://newsletter-builder-11jy.onrender.com (auto-deploys from `main`).

## Pages

- `/` is the full newsletter builder (header, body, photo, shows, merch, footer)
- `/block` is the block generator: just the tour/merch/listen sections, ready to paste into an email

## Running locally

```bash
cd newsletter-builder
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8080/block.

## How the tour map works

Bandsintown supplies exact venue coordinates with each event, and the map is drawn from those directly (no geocoding). The image is served from a stable URL, `/tourmap/<band>.png`, which regenerates from the band's current shows on demand, so map images embedded in sent emails keep working across redeploys. The `?v=<hash>` query string on embedded URLs busts email-client caches when the tour changes.

Note: photos uploaded through the UI are stored on the server's ephemeral disk and will stop resolving after a redeploy unless committed to the repo (`newsletter-builder/uploads/` is gitignored; force-add a photo to persist it).

## File structure

```
newsletter-builder/         # Flask app (Render rootDir)
├── app.py                  # Routes, map generation, block builder
├── builder.py              # Legacy CLI (interactive newsletter build)
├── config.py               # Bands, themes, fonts, API settings
├── scrapers/
│   ├── merch.py            # Merch page scraper
│   └── shows.py            # Bandsintown API (includes venue lat/long)
└── templates/
    ├── web_ui.html         # Full builder UI
    ├── web_ui_block.html   # Block generator UI
    ├── newsletter.html     # Full email template (Jinja2)
    └── newsletter_block.html  # Block template (Jinja2)
```

## Email compatibility

The generated HTML uses table-based layout, inline CSS, web-safe fonts, and MSO conditional comments. Tested in Gmail, Outlook (desktop/web), Apple Mail, and mobile clients.

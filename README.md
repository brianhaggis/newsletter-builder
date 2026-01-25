# House of Hamill Newsletter Builder

A Python tool that generates email-safe HTML newsletters by pulling tour dates from Bandsintown and merch from the House of Hamill website.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Bandsintown API key:**
   - Go to [Bandsintown for Artists](https://artists.bandsintown.com)
   - Navigate to Settings → General → Get API Key
   - Copy the key into `config.py`

## Usage

### Interactive Mode
```bash
python builder.py
```

This will:
1. Fetch upcoming shows from Bandsintown
2. Scrape current merch from the website
3. Prompt you for a subject line
4. Let you choose a merch item to spotlight
5. Open a text editor for your newsletter body
6. Generate the HTML, copy it to clipboard, and open a preview

### Quick Preview
```bash
python builder.py --preview
```

Generates a sample newsletter with placeholder text so you can see the layout.

## File Structure

```
newsletter-builder/
├── builder.py          # Main script
├── config.py           # API keys and settings
├── requirements.txt    # Python dependencies
├── scrapers/
│   ├── merch.py        # Merch page scraper
│   └── shows.py        # Bandsintown API
├── templates/
│   └── newsletter.html # Email template (Jinja2)
└── output/             # Generated newsletters
```

## Email Compatibility

The generated HTML uses:
- Table-based layout (works in Outlook)
- Inline CSS (required for Gmail)
- Web-safe fonts
- Single-column responsive design
- MSO conditional comments for Outlook quirks

Tested in Gmail, Outlook (desktop/web), Apple Mail, and mobile clients.

## Customization

### Colors and Fonts
Edit `config.py` to change brand colors and fonts.

### Template
Edit `templates/newsletter.html` to change the layout. It uses Jinja2 templating.

### Show Count
Change `MAX_SHOWS_TO_DISPLAY` in `config.py` to include more or fewer tour dates.

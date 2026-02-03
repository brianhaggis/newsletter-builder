# Newsletter Builder Configuration

# Default header image for newsletters (landscape banner)
DEFAULT_HEADER_IMAGE = "https://houseofhamill.com/usersite/images/14597053/popup.jpg"

# Bandsintown API
BANDSINTOWN_API_KEY = "bd6845ee17cbcb49d93e3de843d13a21"
ARTIST_NAME = "House of Hamill"

# Website URLs
MERCH_URL = "https://www.houseofhamill.com/merchandise"
MUSIC_URL = "https://www.houseofhamill.com/music"
STORE_URL = "https://www.houseofhamill.com/store"

# Newsletter settings
MAX_SHOWS_TO_DISPLAY = 100  # Show all upcoming shows
NEWSLETTER_WIDTH = 600    # Fluid email width (matches Bandzoogle)

# Color themes for newsletters (all designed for high readability)
# Each theme includes display_font for headings (band name, section titles)
# Google Fonts are used with web-safe fallbacks for email clients that strip them
COLOR_THEMES = {
    "golden": {
        "name": "Golden",
        "header_bg": "#1a1a1a",
        "header_text": "#ffffff",
        "header_subtitle": "#c9a227",
        "accent": "#c9a227",
        "accent_text": "#1a1a1a",
        "body_bg": "#ffffff",
        "body_text": "#333333",
        "secondary_text": "#666666",
        "section_bg": "#f9f5eb",
        "alt_row_bg": "#f9f5eb",
        "footer_bg": "#1a1a1a",
        "footer_text": "#ffffff",
        "merch_bg": "#1a1a1a",
        "merch_text": "#ffffff",
        "listen_bg": "#f8f5eb",
        "ui_bg": "#2d2a25",
        "display_font": "'Playfair Display', Georgia, serif",
        "google_font": "Playfair+Display:wght@400;700",
    },
    "sandy": {
        "name": "Sandy",
        "header_bg": "#5c4a3d",
        "header_text": "#ffffff",
        "header_subtitle": "#d4a574",
        "accent": "#d4a574",
        "accent_text": "#3d2e24",
        "body_bg": "#faf8f5",
        "body_text": "#3d2e24",
        "secondary_text": "#6b5c52",
        "section_bg": "#f0ebe4",
        "alt_row_bg": "#f5f0e8",
        "footer_bg": "#5c4a3d",
        "footer_text": "#ffffff",
        "merch_bg": "#3d2e24",
        "merch_text": "#ffffff",
        "listen_bg": "#f0ebe4",
        "ui_bg": "#8c7a6b",
        "display_font": "'Raleway', Arial, sans-serif",
        "google_font": "Raleway:wght@400;600;700",
    },
    "ocean": {
        "name": "Ocean",
        "header_bg": "#1e3a5f",
        "header_text": "#ffffff",
        "header_subtitle": "#5ba4c9",
        "accent": "#2e8b9a",
        "accent_text": "#ffffff",
        "body_bg": "#ffffff",
        "body_text": "#2c3e50",
        "secondary_text": "#5d6d7e",
        "section_bg": "#e8f4f8",
        "alt_row_bg": "#f0f7fa",
        "footer_bg": "#1e3a5f",
        "footer_text": "#ffffff",
        "merch_bg": "#164050",
        "merch_text": "#ffffff",
        "listen_bg": "#e8f4f8",
        "ui_bg": "#3d6a8a",
        "display_font": "'Cormorant Garamond', Georgia, serif",
        "google_font": "Cormorant+Garamond:wght@400;600;700",
    },
    "forest": {
        "name": "Forest",
        "header_bg": "#2d4a3e",
        "header_text": "#ffffff",
        "header_subtitle": "#8fbc8f",
        "accent": "#5a8f6a",
        "accent_text": "#ffffff",
        "body_bg": "#fafcfa",
        "body_text": "#2d3b2d",
        "secondary_text": "#4a5f4a",
        "section_bg": "#e8f0e8",
        "alt_row_bg": "#f0f5f0",
        "footer_bg": "#2d4a3e",
        "footer_text": "#ffffff",
        "merch_bg": "#1f3328",
        "merch_text": "#ffffff",
        "listen_bg": "#e8f0e8",
        "ui_bg": "#4a7a62",
        "display_font": "'Oswald', Arial, sans-serif",
        "google_font": "Oswald:wght@400;500;700",
    },
    "sunset": {
        "name": "Sunset",
        "header_bg": "#8b4513",
        "header_text": "#ffffff",
        "header_subtitle": "#e8a54c",
        "accent": "#cc5500",
        "accent_text": "#ffffff",
        "body_bg": "#fffbf5",
        "body_text": "#3d2b1f",
        "secondary_text": "#6b5344",
        "section_bg": "#f5ebe0",
        "alt_row_bg": "#faf3ea",
        "footer_bg": "#8b4513",
        "footer_text": "#ffffff",
        "merch_bg": "#5c2e0a",
        "merch_text": "#ffffff",
        "listen_bg": "#f5ebe0",
        "ui_bg": "#a65d2e",
        "display_font": "'EB Garamond', Georgia, serif",
        "google_font": "EB+Garamond:wght@400;600;700",
    },
    "slate": {
        "name": "Slate",
        "header_bg": "#3d4f5f",
        "header_text": "#ffffff",
        "header_subtitle": "#7eb8da",
        "accent": "#4a90a4",
        "accent_text": "#ffffff",
        "body_bg": "#ffffff",
        "body_text": "#363636",
        "secondary_text": "#5a5a5a",
        "section_bg": "#f0f4f7",
        "alt_row_bg": "#f5f8fa",
        "footer_bg": "#3d4f5f",
        "footer_text": "#ffffff",
        "merch_bg": "#2a3a47",
        "merch_text": "#ffffff",
        "listen_bg": "#f0f4f7",
        "ui_bg": "#6a8090",
        "display_font": "'Montserrat', Arial, sans-serif",
        "google_font": "Montserrat:wght@400;600;700",
    },
}

# Default theme
DEFAULT_THEME = "golden"

# Legacy COLORS dict (for backwards compatibility)
COLORS = COLOR_THEMES[DEFAULT_THEME]

# Fonts (web-safe for email)
FONTS = {
    "heading": "Georgia, 'Times New Roman', serif",
    "body": "Arial, Helvetica, sans-serif",
}

# Email settings (configure via environment variables in production)
import os
RESEND_API_KEY = os.environ.get('RESEND_API_KEY', '')
TEST_EMAIL_RECIPIENT = os.environ.get('TEST_EMAIL_RECIPIENT', 'brianhaggis@gmail.com')
EMAIL_FROM = os.environ.get('EMAIL_FROM', 'onboarding@resend.dev')  # Use your verified domain in production

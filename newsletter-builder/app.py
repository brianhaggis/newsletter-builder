#!/usr/bin/env python3
"""
Newsletter Builder Web UI

A Flask-based web interface for building House of Hamill newsletters
with a side-by-side editor and live preview.

Usage:
    python app.py
    Then open http://localhost:5000
"""

import sys
import os
import base64
import uuid
import io
import hashlib
from pathlib import Path
from datetime import datetime
from functools import wraps
import time

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify, Response, send_from_directory, redirect
from jinja2 import Environment, FileSystemLoader
from urllib.parse import quote
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from scrapers.merch import get_all_merch
from scrapers.shows import get_upcoming_shows
from config import (FONTS, DEFAULT_HEADER_IMAGE, COLOR_THEMES, DEFAULT_THEME,
                    RESEND_API_KEY, TEST_EMAIL_RECIPIENT, EMAIL_FROM,
                    BANDS, DEFAULT_BAND)
import requests

app = Flask(__name__)

# Image upload settings
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
MAX_IMAGE_WIDTH = 1400  # 2x retina for 700px display
JPEG_QUALITY = 85

# Simple in-memory cache
_cache = {}
CACHE_TTL = 300  # 5 minutes


def cached(key, ttl=CACHE_TTL):
    """Simple caching decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            if key in _cache:
                value, timestamp = _cache[key]
                if now - timestamp < ttl:
                    return value
            result = func(*args, **kwargs)
            _cache[key] = (result, now)
            return result
        return wrapper
    return decorator


def markdown_to_html(text, color="#333333"):
    """
    Convert simple markdown-ish text to HTML paragraphs.
    Handles line breaks and basic formatting.
    """
    if not text:
        return ""

    # Split into paragraphs (double newline)
    paragraphs = text.strip().split('\n\n')

    html_parts = []
    for p in paragraphs:
        # Replace single newlines with <br>
        p = p.replace('\n', '<br>')
        html_parts.append(f'<p style="margin: 0; text-align: left; color: {color};">{p}</p>')

    return '\n'.join(html_parts)


def get_base_url():
    """Get the base URL for converting relative URLs to absolute."""
    # Check for configured base URL (for production)
    base_url = os.environ.get('BASE_URL', '').rstrip('/')
    if base_url:
        return base_url

    # Check for Render URL
    render_url = os.environ.get('RENDER_EXTERNAL_URL', '').rstrip('/')
    if render_url:
        return render_url

    # Always use production URL for generated HTML (so copied HTML works)
    # This ensures images work when HTML is pasted into Bandzoogle etc.
    return 'https://newsletter-builder-11jy.onrender.com'


def convert_relative_urls(html, base_url):
    """
    Convert relative URLs (like /uploads/...) to absolute URLs.
    This is critical for email compatibility - relative URLs won't work
    when the HTML is pasted into an email client.
    """
    import re

    # Convert src="/uploads/..." to src="https://base/uploads/..."
    html = re.sub(
        r'src="(/uploads/[^"]+)"',
        f'src="{base_url}\\1"',
        html
    )

    # Also handle any other relative URLs in src attributes
    html = re.sub(
        r'src="/([^"]+)"',
        f'src="{base_url}/\\1"',
        html
    )

    return html


def build_newsletter_html(body_text, shows=None, merch=None, photo_url=None, subject="", tour_map_url=None, theme=None, tagline="Upcycled Celtic Folk", include_food_drive=False, include_merch=True, include_listen=True):
    """
    Build the newsletter HTML from components.
    """
    # Set up Jinja
    template_dir = Path(__file__).parent / 'templates'
    env = Environment(loader=FileSystemLoader(template_dir))
    env.filters['urlencode'] = lambda s: quote(str(s), safe='')
    template = env.get_template('newsletter.html')

    # Get theme colors
    if theme and theme in COLOR_THEMES:
        theme_colors = COLOR_THEMES[theme]
    else:
        theme_colors = COLOR_THEMES[DEFAULT_THEME]

    # Check if body is already HTML (from rich text editor) or plain text
    if body_text and body_text.strip().startswith('<'):
        # Already HTML from Quill editor
        import re
        body_html = body_text

        # Get body text color for inline styles
        body_color = theme_colors.get('body_text', '#333333')

        # Remove empty paragraphs that Quill adds (they cause double-spacing)
        body_html = re.sub(r'<p><br></p>', '', body_html)
        body_html = re.sub(r'<p>\s*</p>', '', body_html)
        # Remove trailing <br> at end of paragraphs (Quill adds these)
        body_html = re.sub(r'<br></p>', '</p>', body_html)

        # Convert Quill alignment classes to inline styles (email clients strip CSS classes)
        body_html = re.sub(
            r'class="ql-align-center"',
            f'style="margin: 0; text-align: center; color: {body_color};"',
            body_html
        )
        body_html = re.sub(
            r'class="ql-align-right"',
            f'style="margin: 0; text-align: right; color: {body_color};"',
            body_html
        )
        body_html = re.sub(
            r'class="ql-align-justify"',
            f'style="margin: 0; text-align: justify; color: {body_color};"',
            body_html
        )

        # Handle <p> with existing style attribute - append margin and color
        body_html = re.sub(
            r'<p style="([^"]*)"',
            f'<p style="margin: 0; text-align: left; color: {body_color}; \\1"',
            body_html
        )
        # Handle <p> without any attributes - add full styling
        body_html = re.sub(
            r'<p>(?!</p>)',
            f'<p style="margin: 0; text-align: left; color: {body_color};">',
            body_html
        )

        # Make all links open in new tab (add target="_blank" if not present)
        # First, handle links that already have target attribute
        body_html = re.sub(
            r'<a\s+([^>]*?)target="[^"]*"([^>]*)>',
            r'<a \1target="_blank"\2>',
            body_html
        )
        # Then, add target="_blank" to links without it
        body_html = re.sub(
            r'<a\s+(?![^>]*target=)([^>]*)>',
            r'<a target="_blank" \1>',
            body_html
        )

        # Remove underlines from links (Bandzoogle adds them by default)
        # Add text-decoration: none to all links
        body_html = re.sub(
            r'<a\s+([^>]*)style="([^"]*)"([^>]*)>',
            r'<a \1style="text-decoration: none !important; \2"\3>',
            body_html
        )
        # Handle links without style attribute
        body_html = re.sub(
            r'<a\s+(?![^>]*style=)([^>]*)>',
            r'<a style="text-decoration: none !important;" \1>',
            body_html
        )
    else:
        # Convert plain text to HTML
        body_color = theme_colors.get('body_text', '#333333')
        body_html = markdown_to_html(body_text, body_color)

    # Get base URL for converting relative URLs
    base_url = get_base_url()

    # Convert relative URLs in photo_url and tour_map_url
    if photo_url and photo_url.startswith('/'):
        photo_url = base_url + photo_url
    if tour_map_url and tour_map_url.startswith('/'):
        tour_map_url = base_url + tour_map_url

    # Generate button info (URL + dimensions) for the template
    buttons = {
        'tickets': get_button('TICKETS', theme_colors['accent'], theme_colors['accent_text'], font_size=14, padding_x=20, padding_y=10, show_border=False),
        'see_all_shows': get_button(f"SEE ALL {len(shows or [])} SHOWS", theme_colors['accent'], theme_colors['accent_text'], font_size=16, padding_x=32, padding_y=14),
        'shop_now': get_button('SHOP NOW', theme_colors['accent'], theme_colors['accent_text'], font_size=16, padding_x=28, padding_y=12),
        'spotify': get_button('SPOTIFY', theme_colors['accent'], theme_colors['accent_text'], font_size=14, padding_x=20, padding_y=10),
        'apple': get_button('APPLE', theme_colors['accent'], theme_colors['accent_text'], font_size=14, padding_x=20, padding_y=10),
        'amazon': get_button('AMAZON', theme_colors['accent'], theme_colors['accent_text'], font_size=14, padding_x=20, padding_y=10),
        'youtube': get_button('YOUTUBE', theme_colors['accent'], theme_colors['accent_text'], font_size=14, padding_x=20, padding_y=10),
        'food_drive': get_button('LEARN MORE AND VOLUNTEER', '#ffca28', '#1b5e20', font_size=18, padding_x=36, padding_y=16),
    }

    # Render template
    html = template.render(
        subject=subject,
        tagline=tagline or "Upcycled Celtic Folk",
        body_html=body_html,
        shows=shows or [],
        merch=merch,
        photo_url=photo_url,
        tour_map_url=tour_map_url,
        year=datetime.now().year,
        theme=theme_colors,
        include_food_drive=include_food_drive,
        include_merch=include_merch,
        include_listen=include_listen,
        buttons=buttons,
        get_button=get_button,  # Pass function for dynamic buttons with dimensions
    )

    # Convert any remaining relative URLs in the body HTML (inline images, etc.)
    html = convert_relative_urls(html, base_url)

    return html


@app.route('/')
def index():
    """Main web UI page."""
    return render_template('web_ui.html', default_header_image=DEFAULT_HEADER_IMAGE)


@app.route('/api/bands')
def api_bands():
    """Get available bands."""
    bands_list = []
    for band_id, band_data in BANDS.items():
        bands_list.append({
            "id": band_id,
            "name": band_data["name"],
            "short_name": band_data["short_name"],
            "has_food_drive": band_data.get("has_food_drive", False),
            "has_camp_haggis": band_data.get("has_camp_haggis", False),
        })
    return jsonify({'success': True, 'bands': bands_list, 'default': DEFAULT_BAND})


def get_shows_cached(band_id):
    """Fetch upcoming shows for a band, cached for CACHE_TTL seconds."""
    cache_key = f'shows_{band_id}'
    now = time.time()
    if cache_key in _cache:
        value, timestamp = _cache[cache_key]
        if now - timestamp < CACHE_TTL:
            return value
    shows = get_upcoming_shows(band_id)
    _cache[cache_key] = (shows, now)
    return shows


@app.route('/api/shows')
def api_shows():
    """Fetch tour dates (cached 5 min per band)."""
    band_id = request.args.get('band', DEFAULT_BAND)
    try:
        shows = get_shows_cached(band_id)
        return jsonify({'success': True, 'shows': shows, 'band': band_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'shows': [], 'band': band_id})


@app.route('/api/merch')
def api_merch():
    """Fetch merchandise (cached 5 min per band)."""
    band_id = request.args.get('band', DEFAULT_BAND)
    cache_key = f'merch_{band_id}'

    # Check cache
    now = time.time()
    if cache_key in _cache:
        value, timestamp = _cache[cache_key]
        if now - timestamp < CACHE_TTL:
            products = value
            in_stock = [p for p in products if p['in_stock']]
            return jsonify({'success': True, 'merch': in_stock, 'all_merch': products, 'band': band_id})

    try:
        products = get_all_merch(band_id)
        _cache[cache_key] = (products, now)
        in_stock = [p for p in products if p['in_stock']]
        return jsonify({'success': True, 'merch': in_stock, 'all_merch': products, 'band': band_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'merch': [], 'all_merch': [], 'band': band_id})


@app.route('/api/themes')
def api_themes():
    """Get available color themes with full color data."""
    themes = []
    for theme_id, theme_data in COLOR_THEMES.items():
        themes.append({
            "id": theme_id,
            "name": theme_data["name"],
            "colors": theme_data  # Include full color data for UI theming
        })
    return jsonify({'success': True, 'themes': themes, 'default': DEFAULT_THEME})


@app.route('/api/preview', methods=['POST'])
def api_preview():
    """Generate preview HTML."""
    data = request.get_json()

    subject = data.get('subject', '')
    tagline = data.get('tagline', 'Upcycled Celtic Folk')
    body_text = data.get('body', '')
    photo_url = data.get('photo_url') or None
    merch = data.get('merch') or None
    shows = data.get('shows') or []
    tour_map_url = data.get('tour_map_url') or None
    theme = data.get('theme') or None
    include_food_drive = data.get('include_food_drive', False)
    include_merch = data.get('include_merch', True)
    include_listen = data.get('include_listen', True)

    html = build_newsletter_html(
        body_text=body_text,
        shows=shows,
        merch=merch,
        photo_url=photo_url,
        subject=subject,
        tour_map_url=tour_map_url,
        theme=theme,
        tagline=tagline,
        include_food_drive=include_food_drive,
        include_merch=include_merch,
        include_listen=include_listen
    )

    return jsonify({'success': True, 'html': html})


@app.route('/api/download', methods=['POST'])
def api_download():
    """Download HTML file."""
    data = request.get_json()

    subject = data.get('subject', '')
    tagline = data.get('tagline', 'Upcycled Celtic Folk')
    body_text = data.get('body', '')
    photo_url = data.get('photo_url') or None
    merch = data.get('merch') or None
    shows = data.get('shows') or []
    tour_map_url = data.get('tour_map_url') or None
    theme = data.get('theme') or None
    include_food_drive = data.get('include_food_drive', False)
    include_merch = data.get('include_merch', True)
    include_listen = data.get('include_listen', True)

    html = build_newsletter_html(
        body_text=body_text,
        shows=shows,
        merch=merch,
        photo_url=photo_url,
        subject=subject,
        tour_map_url=tour_map_url,
        theme=theme,
        tagline=tagline,
        include_food_drive=include_food_drive,
        include_merch=include_merch,
        include_listen=include_listen
    )

    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"newsletter_{timestamp}.html"

    return Response(
        html,
        mimetype='text/html',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


@app.route('/api/refresh')
def api_refresh():
    """Clear all caches and refresh data."""
    global _cache, _tour_map_cache
    _cache = {}
    _tour_map_cache = {}
    return jsonify({'success': True, 'message': 'All caches cleared (data + tour maps)'})


@app.route('/api/diagnostic')
def api_diagnostic():
    """Diagnostic endpoint to debug map generation issues."""
    import subprocess

    info = {
        'working_directory': os.getcwd(),
        'app_file': __file__,
        'python_version': sys.version,
    }

    # Check app.py modification time
    try:
        app_stat = os.stat(__file__)
        info['app_modified'] = datetime.fromtimestamp(app_stat.st_mtime).isoformat()
    except:
        info['app_modified'] = 'unknown'

    # Get git commit if available
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info['git_commit'] = result.stdout.strip()
    except:
        info['git_commit'] = 'unknown'

    # Check tour map cache
    info['tour_map_cache_size'] = len(_tour_map_cache)

    return jsonify(info)


@app.route('/api/send-test', methods=['POST'])
def api_send_test():
    """Send a test email with the current newsletter using Resend."""
    # Read API key at request time to pick up env var changes
    api_key = os.environ.get('RESEND_API_KEY', '') or RESEND_API_KEY
    if not api_key:
        return jsonify({
            'success': False,
            'error': 'Email not configured. Set RESEND_API_KEY environment variable.'
        })

    data = request.get_json()

    subject = data.get('subject', 'House of Hamill Newsletter Test')
    tagline = data.get('tagline', 'Upcycled Celtic Folk')
    body_text = data.get('body', '')
    photo_url = data.get('photo_url') or None
    merch = data.get('merch') or None
    shows = data.get('shows') or []
    tour_map_url = data.get('tour_map_url') or None
    theme = data.get('theme') or None
    include_food_drive = data.get('include_food_drive', False)
    include_merch = data.get('include_merch', True)
    include_listen = data.get('include_listen', True)
    recipient = data.get('recipient') or TEST_EMAIL_RECIPIENT

    # Build the HTML
    html_content = build_newsletter_html(
        body_text=body_text,
        shows=shows,
        merch=merch,
        photo_url=photo_url,
        subject=subject,
        tour_map_url=tour_map_url,
        theme=theme,
        tagline=tagline,
        include_food_drive=include_food_drive,
        include_merch=include_merch,
        include_listen=include_listen
    )

    try:
        # Send via Resend API
        response = requests.post(
            'https://api.resend.com/emails',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'from': EMAIL_FROM,
                'to': [recipient],
                'subject': f"[TEST] {subject}" if subject else "[TEST] House of Hamill Newsletter",
                'html': html_content
            }
        )

        if response.status_code == 200:
            return jsonify({
                'success': True,
                'message': f'Test email sent to {recipient}'
            })
        else:
            error_data = response.json()
            return jsonify({
                'success': False,
                'error': error_data.get('message', 'Failed to send email')
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """
    Upload and resize an image.
    Accepts base64 data URL or file upload.
    Returns URL to the resized image.
    """
    try:
        data = request.get_json()
        image_data = data.get('image')  # base64 data URL

        if not image_data:
            return jsonify({'success': False, 'error': 'No image provided'})

        # Parse base64 data URL
        if image_data.startswith('data:'):
            # Extract the base64 part
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
        else:
            return jsonify({'success': False, 'error': 'Invalid image format'})

        # Open with Pillow
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary (for JPEG)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        # Resize if wider than max width
        if img.width > MAX_IMAGE_WIDTH:
            ratio = MAX_IMAGE_WIDTH / img.width
            new_height = int(img.height * ratio)
            img = img.resize((MAX_IMAGE_WIDTH, new_height), Image.LANCZOS)

        # Generate unique filename
        filename = f"{uuid.uuid4().hex[:12]}.jpg"
        filepath = UPLOAD_FOLDER / filename

        # Save as optimized JPEG
        img.save(filepath, 'JPEG', quality=JPEG_QUALITY, optimize=True)

        # Get file size for info
        file_size = filepath.stat().st_size
        size_kb = file_size / 1024

        # Return the URL
        url = f"/uploads/{filename}"
        return jsonify({
            'success': True,
            'url': url,
            'width': img.width,
            'height': img.height,
            'size_kb': round(size_kb, 1)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded images."""
    # Tour maps used to be written here; the files are lost on redeploy
    # (ephemeral disk), so old email URLs get the current map instead
    if filename.startswith('tourmap_') and not (UPLOAD_FOLDER / filename).exists():
        return redirect(f'/tourmap/{DEFAULT_BAND}.png')
    return send_from_directory(UPLOAD_FOLDER, filename)


# Button image generation
BUTTON_CACHE = {}  # Cache generated buttons in memory


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def generate_button_image(text, bg_color, text_color, font_size=16, padding_x=36, padding_y=14, border_radius=4, show_border=True):
    """
    Generate a PNG button image with the given text and colors.
    Renders at 2x resolution for retina/crisp display.
    Features a thick left border accent for visual interest (unless show_border=False).
    Returns the image bytes and dimensions (at 1x for CSS sizing).
    """
    # Create cache key
    cache_key = f"{text}|{bg_color}|{text_color}|{font_size}|{padding_x}|{padding_y}|{border_radius}|{show_border}"
    if cache_key in BUTTON_CACHE:
        return BUTTON_CACHE[cache_key]

    # Convert colors
    bg_rgb = hex_to_rgb(bg_color)
    text_rgb = hex_to_rgb(text_color)

    # Scale factor for retina (2x)
    scale = 2
    scaled_font_size = font_size * scale
    scaled_padding_x = padding_x * scale
    scaled_padding_y = padding_y * scale
    scaled_border_radius = border_radius * scale

    # Left border settings (at 2x scale)
    left_border_width = 10 * scale  # Thick left border

    # Try to load a bold font, fall back to default
    try:
        # Try common system font paths (prefer bold variants)
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux/Ubuntu
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",  # Linux
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",  # Arch Linux
            "C:/Windows/Fonts/arialbd.ttf",  # Windows
            "C:/Windows/Fonts/arial.ttf",    # Windows fallback
        ]
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, scaled_font_size)
                break
        if font is None:
            font = ImageFont.load_default()
            scaled_font_size = font_size  # Default font doesn't scale well
    except Exception:
        font = ImageFont.load_default()

    # Calculate text size
    dummy_img = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate button size (at 2x)
    button_width = text_width + (scaled_padding_x * 2)
    button_height = text_height + (scaled_padding_y * 2)

    # Image size is just the button (no shadow)
    img_width = button_width
    img_height = button_height

    # Create image with transparency
    img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw button background
    draw.rounded_rectangle(
        [(0, 0), (button_width - 1, button_height - 1)],
        radius=scaled_border_radius,
        fill=bg_rgb + (255,)
    )

    # Draw thick left border accent (darker shade of button color)
    # Skip if show_border=False or border_radius is large (causes visual glitches)
    if show_border and border_radius <= 8:
        border_color = tuple(max(0, c - 50) for c in bg_rgb) + (255,)
        draw.rounded_rectangle(
            [(0, 0), (left_border_width, button_height - 1)],
            radius=scaled_border_radius,
            fill=border_color
        )
        # Re-draw main area to clean up where border overlaps
        draw.rounded_rectangle(
            [(left_border_width - 2, 0), (button_width - 1, button_height - 1)],
            radius=scaled_border_radius,
            fill=bg_rgb + (255,)
        )

    # Draw text centered on button
    text_x = (button_width - text_width) // 2
    text_y = (button_height - text_height) // 2 - (bbox[1])
    draw.text((text_x, text_y), text, font=font, fill=text_rgb + (255,))

    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    buf.seek(0)

    # Return 1x dimensions for CSS sizing (image is 2x for retina)
    result = {
        'bytes': buf.getvalue(),
        'width': img_width // scale,
        'height': img_height // scale
    }

    BUTTON_CACHE[cache_key] = result
    return result


@app.route('/api/button')
def api_button():
    """
    Generate a button image on-the-fly.
    Query params: text, bg, fg, size (optional), px (optional), py (optional), r (optional), noborder (optional)
    """
    text = request.args.get('text', 'BUTTON')
    bg_color = request.args.get('bg', '#c9a227')
    text_color = request.args.get('fg', '#1a1a1a')
    font_size = int(request.args.get('size', 16))
    padding_x = int(request.args.get('px', 36))
    padding_y = int(request.args.get('py', 14))
    border_radius = int(request.args.get('r', 4))
    show_border = request.args.get('noborder', '0') != '1'

    try:
        result = generate_button_image(
            text=text,
            bg_color=bg_color,
            text_color=text_color,
            font_size=font_size,
            padding_x=padding_x,
            padding_y=padding_y,
            border_radius=border_radius,
            show_border=show_border
        )

        return Response(
            result['bytes'],
            mimetype='image/png',
            headers={
                'Cache-Control': 'public, max-age=31536000',  # Cache for 1 year
                'Content-Type': 'image/png'
            }
        )
    except Exception as e:
        # Return a 1x1 transparent pixel on error
        return Response(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82',
            mimetype='image/png'
        )


def get_button_url(text, bg_color, text_color, font_size=16, padding_x=36, padding_y=14, border_radius=4, show_border=True):
    """
    Generate the URL for a button image.
    Uses absolute URL so it works in emails.
    """
    from urllib.parse import urlencode
    base_url = get_base_url()
    params = {
        'text': text,
        'bg': bg_color,
        'fg': text_color,
        'size': font_size,
        'px': padding_x,
        'py': padding_y,
        'r': border_radius
    }
    if not show_border:
        params['noborder'] = '1'
    return f"{base_url}/api/button?{urlencode(params)}"


# Subheading image generation
SUBHEADING_CACHE = {}


def generate_subheading_image(text, bg_color, width=570):
    """
    Generate a full-width subheading image with colored background and white text.
    Returns image bytes and dimensions.
    """
    cache_key = f"{text}|{bg_color}|{width}"
    if cache_key in SUBHEADING_CACHE:
        return SUBHEADING_CACHE[cache_key]

    bg_rgb = hex_to_rgb(bg_color)
    text_rgb = (255, 255, 255)  # White text
    # Darker border color (reduce each channel by ~40%)
    border_rgb = tuple(max(0, int(c * 0.6)) for c in bg_rgb)

    scale = 2  # Retina
    scaled_width = width * scale
    font_size = 26
    scaled_font_size = font_size * scale
    padding_x = 15 * scale
    padding_y = 20 * scale  # Taller padding
    border_height = 4 * scale  # Bottom border thickness

    # Load font
    try:
        font_paths = [
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/System/Library/Fonts/Georgia.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, scaled_font_size)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Calculate text size
    dummy_img = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_height = bbox[3] - bbox[1]

    # Calculate image height (including bottom border)
    img_height = text_height + (padding_y * 2) + border_height

    # Create image
    img = Image.new('RGB', (scaled_width, img_height), bg_rgb)
    draw = ImageDraw.Draw(img)

    # Draw darker bottom border
    draw.rectangle(
        [(0, img_height - border_height), (scaled_width, img_height)],
        fill=border_rgb
    )

    # Draw text (left-aligned with padding)
    text_y = padding_y - bbox[1]
    draw.text((padding_x, text_y), text, font=font, fill=text_rgb)

    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    buf.seek(0)

    result = {
        'bytes': buf.getvalue(),
        'width': scaled_width // scale,
        'height': img_height // scale
    }

    SUBHEADING_CACHE[cache_key] = result
    return result


@app.route('/api/subheading')
def api_subheading():
    """Generate a subheading image on-the-fly."""
    text = request.args.get('text', 'Subheading')
    bg_color = request.args.get('bg', '#c9a227')
    width = int(request.args.get('width', 634))  # Full width inside border (650 - 16)

    try:
        result = generate_subheading_image(text, bg_color, width)
        return Response(
            result['bytes'],
            mimetype='image/png',
            headers={
                'Cache-Control': 'public, max-age=31536000',
                'Content-Type': 'image/png'
            }
        )
    except Exception as e:
        print(f"Subheading generation error: {e}")
        return Response(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82',
            mimetype='image/png'
        )


def get_button(text, bg_color, text_color, font_size=16, padding_x=36, padding_y=14, border_radius=4, show_border=True):
    """
    Generate button info including URL and dimensions.
    Returns a dict with 'url', 'width', 'height'.
    """
    # Generate the button to get dimensions
    result = generate_button_image(text, bg_color, text_color, font_size, padding_x, padding_y, border_radius, show_border)

    return {
        'url': get_button_url(text, bg_color, text_color, font_size, padding_x, padding_y, border_radius, show_border),
        'width': result['width'],
        'height': result['height']
    }


# Cache for generated tour maps (keyed by sorted coordinate tuple)
_tour_map_cache = {}


def generate_tour_map_simple(coords):
    """
    Generate a US map with state lines using matplotlib.
    Uses simplified state boundary coordinates for a clean look.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f9f5eb')
    ax.set_facecolor('#f9f5eb')

    # Set extent to continental US
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)

    # US continental outline
    us_outline = [
        (-124.7, 48.4), (-124.6, 48.0), (-124.0, 46.3), (-123.5, 46.0),
        (-124.0, 44.5), (-124.5, 43.0), (-124.4, 42.0), (-124.2, 41.0),
        (-124.0, 40.0), (-123.5, 39.0), (-122.5, 38.0), (-122.0, 37.0),
        (-121.5, 36.5), (-120.5, 35.5), (-120.0, 34.5), (-119.0, 34.0),
        (-118.5, 34.0), (-117.5, 33.5), (-117.1, 32.6),
        (-114.7, 32.7), (-111.1, 31.4), (-108.2, 31.4), (-106.5, 31.8),
        (-104.9, 29.6), (-104.0, 29.5), (-103.1, 29.0), (-102.4, 29.8),
        (-101.0, 29.8), (-100.0, 28.7), (-99.2, 27.0), (-97.5, 26.0),
        (-97.2, 26.5), (-97.0, 27.5), (-96.5, 28.5), (-95.0, 29.0),
        (-94.5, 29.5), (-93.5, 29.7), (-92.0, 29.5), (-91.0, 29.2),
        (-89.5, 29.2), (-89.0, 29.0), (-88.5, 30.2), (-88.0, 30.2),
        (-87.5, 30.3), (-86.5, 30.4), (-85.5, 30.0), (-85.0, 29.5),
        (-84.0, 29.5), (-83.5, 29.0), (-82.5, 27.5), (-81.5, 25.5),
        (-80.5, 25.2), (-80.0, 25.8),
        (-80.0, 26.5), (-80.2, 28.0), (-81.0, 29.5), (-81.5, 30.7),
        (-81.0, 31.5), (-80.5, 32.0), (-79.5, 33.0), (-78.5, 34.0),
        (-77.5, 34.5), (-76.0, 35.0), (-75.5, 35.5), (-75.5, 36.5),
        (-76.0, 37.0), (-76.3, 37.5), (-76.0, 38.0), (-75.5, 38.5),
        (-75.2, 39.0), (-75.0, 39.5), (-74.8, 39.8), (-74.2, 40.5),
        (-74.0, 40.7), (-73.8, 41.0), (-73.5, 41.0), (-72.5, 41.0),
        (-71.5, 41.3), (-71.0, 41.5), (-70.5, 41.8), (-70.0, 42.0),
        (-69.8, 43.0), (-69.0, 43.8), (-68.5, 44.3), (-67.5, 44.6),
        (-67.0, 45.0), (-67.0, 47.3),
        (-69.0, 47.4), (-70.0, 46.3), (-70.5, 45.5), (-71.5, 45.0),
        (-73.0, 45.0), (-74.5, 45.0), (-75.0, 44.8), (-76.5, 44.0),
        (-77.5, 43.8), (-79.0, 43.5), (-79.5, 43.2), (-82.0, 43.0),
        (-82.5, 45.0), (-83.5, 46.0), (-84.5, 46.5), (-85.0, 46.8),
        (-86.5, 46.5), (-87.0, 46.5), (-88.0, 48.0), (-89.0, 48.0),
        (-90.0, 48.0), (-95.0, 49.0), (-100.0, 49.0), (-105.0, 49.0),
        (-110.0, 49.0), (-117.0, 49.0), (-123.0, 49.0), (-124.7, 48.4)
    ]

    # Draw US fill
    xs, ys = zip(*us_outline)
    ax.fill(xs, ys, color='#e8e0d0', edgecolor='#888888', linewidth=1.0, zorder=1)

    # Simplified state boundaries (major lines only for clean look)
    state_lines = [
        # West Coast states
        [(-124.2, 46.0), (-117.0, 46.0)],  # WA/OR
        [(-124.2, 42.0), (-117.0, 42.0)],  # OR/CA
        [(-120.0, 39.0), (-114.0, 35.0)],  # CA/NV diagonal
        # Mountain states vertical
        [(-117.0, 49.0), (-117.0, 42.0)],  # WA/ID, OR/ID
        [(-117.0, 42.0), (-114.0, 42.0), (-114.0, 37.0)],  # ID/NV/UT/AZ
        [(-111.0, 49.0), (-111.0, 45.0), (-111.0, 41.0), (-111.0, 37.0), (-111.0, 31.4)],  # MT/ID, WY, UT/CO, AZ/NM
        [(-109.0, 49.0), (-109.0, 45.0), (-109.0, 41.0), (-109.0, 37.0), (-109.0, 31.4)],  # MT/ND border, WY/MT, CO/UT, AZ/NM
        [(-104.0, 49.0), (-104.0, 45.0), (-104.0, 41.0), (-104.0, 37.0), (-104.0, 32.0)],  # ND/MT, SD/WY, NE/CO, CO/NM/TX
        # Horizontal mountain/plains
        [(-117.0, 42.0), (-111.0, 42.0)],  # ID/NV/UT
        [(-111.0, 41.0), (-104.0, 41.0)],  # WY/UT/CO
        [(-109.0, 37.0), (-103.0, 37.0)],  # CO/NM
        [(-111.0, 45.0), (-104.0, 45.0)],  # MT/WY
        # Great Plains vertical
        [(-100.0, 49.0), (-100.0, 40.0)],  # ND/SD/NE
        [(-97.0, 49.0), (-97.0, 43.5)],  # MN/ND/SD
        [(-96.5, 43.5), (-96.5, 40.0)],  # MN/SD/IA/NE
        # Central horizontal
        [(-104.0, 43.0), (-96.5, 43.0)],  # SD/NE
        [(-104.0, 40.0), (-95.3, 40.0)],  # CO/NE/KS
        [(-102.0, 36.5), (-94.5, 36.5)],  # OK/TX panhandle, OK/KS
        # Texas borders
        [(-103.0, 32.0), (-94.0, 32.0)],  # TX/LA (partial)
        [(-100.0, 36.5), (-100.0, 34.5), (-99.0, 34.5)],  # TX panhandle
        # Midwest
        [(-91.5, 49.0), (-91.5, 43.5)],  # MN/WI
        [(-90.5, 43.0), (-90.5, 36.5)],  # WI/IL, IL/MO
        [(-87.5, 42.5), (-87.5, 37.0)],  # IL/IN
        [(-85.0, 42.0), (-85.0, 38.0)],  # IN/OH, KY line
        [(-91.0, 36.5), (-89.0, 36.5)],  # MO/AR
        [(-94.5, 36.5), (-94.5, 33.0)],  # MO/OK/AR
        # Southern horizontal
        [(-88.0, 35.0), (-81.0, 35.0)],  # TN/AL/GA, TN/NC
        [(-88.5, 31.0), (-85.0, 31.0)],  # MS/LA, AL/FL
        [(-85.0, 35.0), (-85.0, 31.0)],  # AL/GA
        [(-82.0, 35.0), (-82.0, 32.0)],  # GA/SC
        # East Coast
        [(-80.5, 35.0), (-75.5, 35.0)],  # NC/SC partial
        [(-83.5, 35.0), (-81.0, 35.0), (-79.0, 36.5)],  # TN/NC/VA
        [(-78.0, 39.5), (-75.5, 39.5)],  # MD/PA
        [(-80.5, 40.5), (-75.0, 40.5)],  # PA/MD/WV
        [(-79.5, 42.0), (-75.0, 42.0)],  # NY/PA
        [(-73.3, 45.0), (-73.3, 42.0)],  # VT/NY
        [(-72.5, 42.0), (-71.0, 42.0)],  # MA/CT
        # Great Lakes region
        [(-84.5, 46.5), (-82.5, 46.0)],  # Upper MI
        [(-87.5, 45.5), (-87.5, 42.5)],  # WI/MI (Lake Michigan)
    ]

    # Draw state lines
    for line in state_lines:
        if len(line) >= 2:
            xs, ys = zip(*line)
            ax.plot(xs, ys, color='#aaaaaa', linewidth=0.5, zorder=2)

    # Plot show locations
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    # Draw glow effect
    ax.scatter(lons, lats, c='#c9a227', s=350, zorder=4, alpha=0.25)
    # Draw main dots
    ax.scatter(lons, lats, c='#c9a227', s=150, zorder=5,
               edgecolors='#1a1a1a', linewidths=1.5, alpha=0.95)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save to bytes (reduced DPI for faster generation)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150,
                facecolor='#f9f5eb', edgecolor='none', pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue()


def coords_from_shows(shows):
    """
    Unique (lat, lon) pairs from show data.
    Coordinates come from the Bandsintown venue record, so shows without
    them (older cached payloads) are simply skipped.
    """
    coords = set()
    for show in shows:
        lat = show.get('latitude')
        lon = show.get('longitude')
        if lat is not None and lon is not None:
            try:
                coords.add((round(float(lat), 3), round(float(lon), 3)))
            except (TypeError, ValueError):
                continue
    return sorted(coords)


def coords_hash(coords):
    """Short stable hash of a coordinate list, used to cache-bust map URLs."""
    return hashlib.md5(repr(coords).encode()).hexdigest()[:12]


def generate_tour_map(coords):
    """
    Generate a US map PNG with a dot for each coordinate.
    Uses caching to avoid regenerating the same map.
    """
    if not coords:
        return None

    cache_key = tuple(coords)
    if cache_key in _tour_map_cache:
        return _tour_map_cache[cache_key]

    print(f"Generating tour map for {len(coords)} locations...")
    map_data = generate_tour_map_simple(coords)
    _tour_map_cache[cache_key] = map_data
    return map_data


STATIC_FOLDER = Path(__file__).parent / 'static'


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(STATIC_FOLDER, filename)


@app.route('/tourmap/<band_id>.png')
def serve_tour_map(band_id):
    """
    Stable tour map URL for embedding in emails.
    Regenerates from the band's current shows on demand, so the image
    survives redeploys (uploads/ is ephemeral on Render).
    """
    if band_id not in BANDS:
        return Response('Unknown band', status=404)

    try:
        shows = get_shows_cached(band_id)
    except Exception as e:
        print(f"Tour map: failed to fetch shows for {band_id}: {e}")
        return Response('Could not fetch shows', status=503)

    map_data = generate_tour_map(coords_from_shows(shows))
    if not map_data:
        return Response('No mappable shows', status=404)

    # Emails append ?v=<hash> for cache busting, so a long max-age is safe
    return Response(map_data, mimetype='image/png',
                    headers={'Cache-Control': 'public, max-age=86400'})


@app.route('/api/tour-map', methods=['POST'])
def api_tour_map():
    """
    Get the tour map image URL for the current shows.
    Returns the stable /tourmap/<band>.png route with a cache-busting
    version hash, and pre-renders the map to validate the data.
    """
    try:
        data = request.get_json() or {}
        band_id = data.get('band') or DEFAULT_BAND
        if band_id not in BANDS:
            return jsonify({'success': False, 'error': f'Unknown band: {band_id}'})

        shows = data.get('shows') or get_shows_cached(band_id)
        coords = coords_from_shows(shows)

        map_data = generate_tour_map(coords)
        if not map_data:
            return jsonify({'success': False, 'error': 'Could not generate map - no valid locations'})

        url = f"/tourmap/{band_id}.png?v={coords_hash(coords)}"
        absolute_url = f"{get_base_url()}{url}"

        return jsonify({'success': True, 'url': url, 'absolute_url': absolute_url,
                        'source': 'dynamic', 'locations': len(coords)})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# BLOCK GENERATOR (Simplified - no header/body/footer)
# ============================================================

def build_block_html(shows=None, merch=None, tour_map_url=None, theme=None, include_food_drive=False, include_camp_haggis=False, band_id=None, include_merch=True, include_listen=True):
    """
    Build just the newsletter block HTML (food drive, tours, merch, listen links).
    No outer wrapper, header, body text, or footer.
    """
    # Set up Jinja
    template_dir = Path(__file__).parent / 'templates'
    env = Environment(loader=FileSystemLoader(template_dir))
    env.filters['urlencode'] = lambda s: quote(str(s), safe='')
    template = env.get_template('newsletter_block.html')

    # Get theme colors
    if theme and theme in COLOR_THEMES:
        theme_colors = COLOR_THEMES[theme]
    else:
        theme_colors = COLOR_THEMES[DEFAULT_THEME]

    # Get band config
    band = BANDS.get(band_id, BANDS[DEFAULT_BAND])

    # Get base URL for absolute URLs
    base_url = get_base_url()

    # Convert relative tour_map_url to absolute
    if tour_map_url and tour_map_url.startswith('/'):
        tour_map_url = base_url + tour_map_url

    # Generate button info
    buttons = {
        'tickets': get_button('TICKETS', theme_colors['accent'], theme_colors['accent_text'], font_size=11, padding_x=14, padding_y=6, show_border=False),
        'shop_now': get_button('SHOP NOW', theme_colors['accent'], theme_colors['accent_text'], font_size=14, padding_x=24, padding_y=10),
        'spotify': get_button('SPOTIFY', theme_colors['accent'], theme_colors['accent_text'], font_size=14, padding_x=20, padding_y=10),
        'apple': get_button('APPLE', theme_colors['accent'], theme_colors['accent_text'], font_size=14, padding_x=20, padding_y=10),
        'amazon': get_button('AMAZON', theme_colors['accent'], theme_colors['accent_text'], font_size=14, padding_x=20, padding_y=10),
        'youtube': get_button('YOUTUBE', theme_colors['accent'], theme_colors['accent_text'], font_size=14, padding_x=20, padding_y=10),
        'food_drive': get_button('LEARN MORE AND VOLUNTEER', '#ffca28', '#1b5e20', font_size=16, padding_x=32, padding_y=14),
        'camp_haggis': get_button('CAMP HAGGIS V: LEARN MORE', '#1565c0', '#ffffff', font_size=16, padding_x=32, padding_y=14, border_radius=20, show_border=False),
    }

    # "See All Shows" button (dynamic based on show count)
    see_all_btn = get_button(f"SEE ALL {len(shows or [])} SHOWS", theme_colors['accent'], theme_colors['accent_text'], font_size=14, padding_x=28, padding_y=12)

    # Render template
    html = template.render(
        shows=shows or [],
        merch=merch,
        tour_map_url=tour_map_url,
        theme=theme_colors,
        include_food_drive=include_food_drive,
        include_camp_haggis=include_camp_haggis,
        include_merch=include_merch,
        include_listen=include_listen,
        buttons=buttons,
        see_all_btn=see_all_btn,
        band=band,  # Pass band config for URLs
    )

    return html


@app.route('/block')
def block_generator():
    """Simplified block generator UI."""
    return render_template('web_ui_block.html')


@app.route('/api/preview-block', methods=['POST'])
def api_preview_block():
    """Generate block preview HTML."""
    data = request.get_json()

    merch = data.get('merch') or None
    shows = data.get('shows') or []
    tour_map_url = data.get('tour_map_url') or None
    theme = data.get('theme') or None
    include_food_drive = data.get('include_food_drive', False)
    include_camp_haggis = data.get('include_camp_haggis', False)
    include_merch = data.get('include_merch', True)
    include_listen = data.get('include_listen', True)
    band_id = data.get('band') or DEFAULT_BAND

    html = build_block_html(
        shows=shows,
        merch=merch,
        tour_map_url=tour_map_url,
        theme=theme,
        include_food_drive=include_food_drive,
        include_camp_haggis=include_camp_haggis,
        band_id=band_id,
        include_merch=include_merch,
        include_listen=include_listen
    )

    return jsonify({'success': True, 'html': html})


if __name__ == '__main__':
    # Use PORT environment variable for cloud hosting (Render, etc.)
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'

    print("\n" + "="*50)
    print("  HOUSE OF HAMILL NEWSLETTER BUILDER")
    print(f"  Web UI running at http://localhost:{port}")
    print(f"  Block Generator at http://localhost:{port}/block")
    print("="*50 + "\n")
    app.run(debug=debug, host='0.0.0.0', port=port)

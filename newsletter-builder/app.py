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

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from jinja2 import Environment, FileSystemLoader
from urllib.parse import quote
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Try to import cartopy for high-quality maps, fall back to simple map
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Cartopy not available, using fallback map rendering")

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Cloudinary for tour map hosting
try:
    import cloudinary
    import cloudinary.uploader
    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False
    print("Cloudinary not available")

from scrapers.merch import get_all_merch
from scrapers.shows import get_upcoming_shows
from config import (FONTS, DEFAULT_HEADER_IMAGE, COLOR_THEMES, DEFAULT_THEME,
                    RESEND_API_KEY, TEST_EMAIL_RECIPIENT, EMAIL_FROM,
                    CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET)

# Configure Cloudinary if credentials are available
if CLOUDINARY_AVAILABLE and CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET
    )
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


def build_newsletter_html(body_text, shows=None, merch=None, photo_url=None, subject="", tour_map_url=None, theme=None, tagline="Upcycled Celtic Folk", include_food_drive=False):
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
        'tickets': get_button('TICKETS', theme_colors['accent'], theme_colors['accent_text'], font_size=14, padding_x=20, padding_y=10),
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


@app.route('/api/shows')
@cached('shows')
def api_shows():
    """Fetch tour dates (cached 5 min)."""
    try:
        shows = get_upcoming_shows()
        return jsonify({'success': True, 'shows': shows})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'shows': []})


@app.route('/api/merch')
@cached('merch')
def api_merch():
    """Fetch merchandise (cached 5 min)."""
    try:
        products = get_all_merch()
        in_stock = [p for p in products if p['in_stock']]
        return jsonify({'success': True, 'merch': in_stock, 'all_merch': products})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'merch': [], 'all_merch': []})


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

    html = build_newsletter_html(
        body_text=body_text,
        shows=shows,
        merch=merch,
        photo_url=photo_url,
        subject=subject,
        tour_map_url=tour_map_url,
        theme=theme,
        tagline=tagline,
        include_food_drive=include_food_drive
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

    html = build_newsletter_html(
        body_text=body_text,
        shows=shows,
        merch=merch,
        photo_url=photo_url,
        subject=subject,
        tour_map_url=tour_map_url,
        theme=theme,
        tagline=tagline,
        include_food_drive=include_food_drive
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
        'cartopy_available': CARTOPY_AVAILABLE,
        'working_directory': os.getcwd(),
        'app_file': __file__,
        'python_version': sys.version,
    }

    # Get cartopy version if available
    if CARTOPY_AVAILABLE:
        import cartopy
        info['cartopy_version'] = cartopy.__version__

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
        include_food_drive=include_food_drive
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
    return send_from_directory(UPLOAD_FOLDER, filename)


# Button image generation
BUTTON_CACHE = {}  # Cache generated buttons in memory


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def generate_button_image(text, bg_color, text_color, font_size=16, padding_x=36, padding_y=14, border_radius=4):
    """
    Generate a PNG button image with the given text and colors.
    Renders at 2x resolution for retina/crisp display.
    Features a thick left border accent for visual interest.
    Returns the image bytes and dimensions (at 1x for CSS sizing).
    """
    # Create cache key
    cache_key = f"{text}|{bg_color}|{text_color}|{font_size}|{padding_x}|{padding_y}|{border_radius}"
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
    Query params: text, bg, fg, size (optional), px (optional), py (optional), r (optional)
    """
    text = request.args.get('text', 'BUTTON')
    bg_color = request.args.get('bg', '#c9a227')
    text_color = request.args.get('fg', '#1a1a1a')
    font_size = int(request.args.get('size', 16))
    padding_x = int(request.args.get('px', 36))
    padding_y = int(request.args.get('py', 14))
    border_radius = int(request.args.get('r', 4))

    try:
        result = generate_button_image(
            text=text,
            bg_color=bg_color,
            text_color=text_color,
            font_size=font_size,
            padding_x=padding_x,
            padding_y=padding_y,
            border_radius=border_radius
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


def get_button_url(text, bg_color, text_color, font_size=16, padding_x=36, padding_y=14, border_radius=4):
    """
    Generate the URL for a button image.
    Uses absolute URL so it works in emails.
    """
    from urllib.parse import urlencode
    base_url = get_base_url()
    params = urlencode({
        'text': text,
        'bg': bg_color,
        'fg': text_color,
        'size': font_size,
        'px': padding_x,
        'py': padding_y,
        'r': border_radius
    })
    return f"{base_url}/api/button?{params}"


def get_button(text, bg_color, text_color, font_size=16, padding_x=36, padding_y=14, border_radius=4):
    """
    Generate button info including URL and dimensions.
    Returns a dict with 'url', 'width', 'height'.
    """
    # Generate the button to get dimensions
    result = generate_button_image(text, bg_color, text_color, font_size, padding_x, padding_y, border_radius)

    return {
        'url': get_button_url(text, bg_color, text_color, font_size, padding_x, padding_y, border_radius),
        'width': result['width'],
        'height': result['height']
    }


# Common US city coordinates lookup table (instant, no API needed)
US_CITY_COORDS = {
    # Alabama
    "birmingham, al": (33.5207, -86.8025), "huntsville, al": (34.7304, -86.5861),
    "mobile, al": (30.6954, -88.0399), "montgomery, al": (32.3792, -86.3077),
    "auburn, al": (32.6099, -85.4808), "tuscaloosa, al": (33.2098, -87.5692),
    # Arizona
    "phoenix, az": (33.4484, -112.0740), "tucson, az": (32.2226, -110.9747),
    "scottsdale, az": (33.4942, -111.9261), "tempe, az": (33.4255, -111.9400),
    # Arkansas
    "little rock, ar": (34.7465, -92.2896), "fayetteville, ar": (36.0626, -94.1574),
    # California
    "los angeles, ca": (34.0522, -118.2437), "san francisco, ca": (37.7749, -122.4194),
    "san diego, ca": (32.7157, -117.1611), "san jose, ca": (37.3382, -121.8863),
    "oakland, ca": (37.8044, -122.2712), "sacramento, ca": (38.5816, -121.4944),
    "santa barbara, ca": (34.4208, -119.6982), "berkeley, ca": (37.8716, -122.2727),
    "pasadena, ca": (34.1478, -118.1445), "santa cruz, ca": (36.9741, -122.0308),
    "anaheim, ca": (33.8366, -117.9143), "fresno, ca": (36.7378, -119.7871),
    # Colorado
    "denver, co": (39.7392, -104.9903), "boulder, co": (40.0150, -105.2705),
    "colorado springs, co": (38.8339, -104.8214), "fort collins, co": (40.5853, -105.0844),
    # Connecticut
    "hartford, ct": (41.7658, -72.6734), "new haven, ct": (41.3083, -72.9279),
    "stamford, ct": (41.0534, -73.5387), "bridgeport, ct": (41.1792, -73.1894),
    # Delaware
    "wilmington, de": (39.7391, -75.5398), "dover, de": (39.1582, -75.5244),
    # Florida
    "miami, fl": (25.7617, -80.1918), "orlando, fl": (28.5383, -81.3792),
    "tampa, fl": (27.9506, -82.4572), "jacksonville, fl": (30.3322, -81.6557),
    "st. petersburg, fl": (27.7676, -82.6403), "fort lauderdale, fl": (26.1224, -80.1373),
    "gainesville, fl": (29.6516, -82.3248), "tallahassee, fl": (30.4383, -84.2807),
    # Georgia
    "atlanta, ga": (33.7490, -84.3880), "savannah, ga": (32.0809, -81.0912),
    "athens, ga": (33.9519, -83.3576), "augusta, ga": (33.4735, -82.0105),
    # Idaho
    "boise, id": (43.6150, -116.2023),
    # Illinois
    "chicago, il": (41.8781, -87.6298), "springfield, il": (39.7817, -89.6501),
    "champaign, il": (40.1164, -88.2434), "evanston, il": (42.0451, -87.6877),
    # Indiana
    "indianapolis, in": (39.7684, -86.1581), "bloomington, in": (39.1653, -86.5264),
    "fort wayne, in": (41.0793, -85.1394), "south bend, in": (41.6764, -86.2520),
    # Iowa
    "des moines, ia": (41.5868, -93.6250), "iowa city, ia": (41.6611, -91.5302),
    "cedar rapids, ia": (41.9779, -91.6656),
    # Kansas
    "kansas city, ks": (39.1141, -94.6275), "wichita, ks": (37.6872, -97.3301),
    "lawrence, ks": (38.9717, -95.2353),
    # Kentucky
    "louisville, ky": (38.2527, -85.7585), "lexington, ky": (38.0406, -84.5037),
    # Louisiana
    "new orleans, la": (29.9511, -90.0715), "baton rouge, la": (30.4515, -91.1871),
    # Maine
    "portland, me": (43.6591, -70.2568), "bangor, me": (44.8016, -68.7712),
    # Maryland
    "baltimore, md": (39.2904, -76.6122), "annapolis, md": (38.9784, -76.4922),
    "bethesda, md": (38.9847, -77.0947), "silver spring, md": (38.9907, -77.0261),
    # Massachusetts
    "boston, ma": (42.3601, -71.0589), "cambridge, ma": (42.3736, -71.1097),
    "worcester, ma": (42.2626, -71.8023), "springfield, ma": (42.1015, -72.5898),
    "northampton, ma": (42.3251, -72.6412), "somerville, ma": (42.3876, -71.0995),
    # Michigan
    "detroit, mi": (42.3314, -83.0458), "ann arbor, mi": (42.2808, -83.7430),
    "grand rapids, mi": (42.9634, -85.6681), "lansing, mi": (42.7325, -84.5555),
    # Minnesota
    "minneapolis, mn": (44.9778, -93.2650), "st. paul, mn": (44.9537, -93.0900),
    "duluth, mn": (46.7867, -92.1005),
    # Mississippi
    "jackson, ms": (32.2988, -90.1848), "oxford, ms": (34.3665, -89.5192),
    # Missouri
    "st. louis, mo": (38.6270, -90.1994), "kansas city, mo": (39.0997, -94.5786),
    "columbia, mo": (38.9517, -92.3341),
    # Montana
    "missoula, mt": (46.8721, -113.9940), "bozeman, mt": (45.6770, -111.0429),
    # Nebraska
    "omaha, ne": (41.2565, -95.9345), "lincoln, ne": (40.8258, -96.6852),
    # Nevada
    "las vegas, nv": (36.1699, -115.1398), "reno, nv": (39.5296, -119.8138),
    # New Hampshire
    "manchester, nh": (42.9956, -71.4548), "portsmouth, nh": (43.0718, -70.7626),
    # New Jersey
    "newark, nj": (40.7357, -74.1724), "jersey city, nj": (40.7178, -74.0431),
    "atlantic city, nj": (39.3643, -74.4229), "hoboken, nj": (40.7440, -74.0324),
    "princeton, nj": (40.3573, -74.6672), "asbury park, nj": (40.2204, -74.0121),
    # New Mexico
    "albuquerque, nm": (35.0844, -106.6504), "santa fe, nm": (35.6870, -105.9378),
    # New York
    "new york, ny": (40.7128, -74.0060), "brooklyn, ny": (40.6782, -73.9442),
    "buffalo, ny": (42.8864, -78.8784), "albany, ny": (42.6526, -73.7562),
    "rochester, ny": (43.1566, -77.6088), "syracuse, ny": (43.0481, -76.1474),
    "ithaca, ny": (42.4440, -76.5019), "poughkeepsie, ny": (41.7004, -73.9210),
    # North Carolina
    "charlotte, nc": (35.2271, -80.8431), "raleigh, nc": (35.7796, -78.6382),
    "durham, nc": (35.9940, -78.8986), "asheville, nc": (35.5951, -82.5515),
    "chapel hill, nc": (35.9132, -79.0558), "greensboro, nc": (36.0726, -79.7920),
    # North Dakota
    "fargo, nd": (46.8772, -96.7898),
    # Ohio
    "columbus, oh": (39.9612, -82.9988), "cleveland, oh": (41.4993, -81.6944),
    "cincinnati, oh": (39.1031, -84.5120), "athens, oh": (39.3292, -82.1013),
    "dayton, oh": (39.7589, -84.1916), "toledo, oh": (41.6528, -83.5379),
    # Oklahoma
    "oklahoma city, ok": (35.4676, -97.5164), "tulsa, ok": (36.1540, -95.9928),
    "norman, ok": (35.2226, -97.4395),
    # Oregon
    "portland, or": (45.5152, -122.6784), "eugene, or": (44.0521, -123.0868),
    "bend, or": (44.0582, -121.3153), "salem, or": (44.9429, -123.0351),
    # Pennsylvania
    "philadelphia, pa": (39.9526, -75.1652), "pittsburgh, pa": (40.4406, -79.9959),
    "state college, pa": (40.7934, -77.8600), "harrisburg, pa": (40.2732, -76.8867),
    "lancaster, pa": (40.0379, -76.3055), "allentown, pa": (40.6084, -75.4902),
    # Rhode Island
    "providence, ri": (41.8240, -71.4128), "newport, ri": (41.4901, -71.3128),
    # South Carolina
    "charleston, sc": (32.7765, -79.9311), "columbia, sc": (34.0007, -81.0348),
    "greenville, sc": (34.8526, -82.3940),
    # South Dakota
    "sioux falls, sd": (43.5446, -96.7311),
    # Tennessee
    "nashville, tn": (36.1627, -86.7816), "memphis, tn": (35.1495, -90.0490),
    "knoxville, tn": (35.9606, -83.9207), "chattanooga, tn": (35.0456, -85.3097),
    # Texas
    "houston, tx": (29.7604, -95.3698), "austin, tx": (30.2672, -97.7431),
    "dallas, tx": (32.7767, -96.7970), "san antonio, tx": (29.4241, -98.4936),
    "fort worth, tx": (32.7555, -97.3308), "el paso, tx": (31.7619, -106.4850),
    "denton, tx": (33.2148, -97.1331),
    # Utah
    "salt lake city, ut": (40.7608, -111.8910), "provo, ut": (40.2338, -111.6585),
    # Vermont
    "burlington, vt": (44.4759, -73.2121), "montpelier, vt": (44.2601, -72.5754),
    # Virginia
    "richmond, va": (37.5407, -77.4360), "norfolk, va": (36.8508, -76.2859),
    "charlottesville, va": (38.0293, -78.4767), "virginia beach, va": (36.8529, -75.9780),
    "arlington, va": (38.8816, -77.0910), "alexandria, va": (38.8048, -77.0469),
    # Washington
    "seattle, wa": (47.6062, -122.3321), "spokane, wa": (47.6588, -117.4260),
    "tacoma, wa": (47.2529, -122.4443), "olympia, wa": (47.0379, -122.9007),
    # Washington DC
    "washington, dc": (38.9072, -77.0369),
    # West Virginia
    "charleston, wv": (38.3498, -81.6326), "morgantown, wv": (39.6295, -79.9559),
    # Wisconsin
    "milwaukee, wi": (43.0389, -87.9065), "madison, wi": (43.0731, -89.4012),
    # Wyoming
    "cheyenne, wy": (41.1400, -104.8202), "jackson, wy": (43.4799, -110.7624),
}

def geocode_location(location_str):
    """
    Convert a location string to lat/lon coordinates.
    Uses built-in lookup table first, falls back to Nominatim.
    """
    if not location_str:
        return None

    # Normalize the location string
    loc_lower = location_str.lower().strip()

    # Check our built-in lookup table first
    if loc_lower in US_CITY_COORDS:
        return US_CITY_COORDS[loc_lower]

    # Try partial matching (e.g., "Nashville, TN" -> "nashville, tn")
    for key, coords in US_CITY_COORDS.items():
        if key.split(',')[0] in loc_lower or loc_lower.split(',')[0].strip() in key:
            return coords

    # Fall back to geocoding API for unknown locations
    if location_str in _geocode_cache:
        return _geocode_cache[location_str]

    try:
        geolocator = Nominatim(user_agent="newsletter-builder-hoh")
        location = geolocator.geocode(location_str, timeout=2)
        if location:
            coords = (location.latitude, location.longitude)
            _geocode_cache[location_str] = coords
            return coords
    except (GeocoderTimedOut, Exception) as e:
        print(f"Geocoding error for {location_str}: {e}")

    _geocode_cache[location_str] = None
    return None


# Runtime geocoding cache for API fallback
_geocode_cache = {}

# Cache for generated tour maps (keyed by sorted coordinate tuple)
_tour_map_cache = {}


def generate_tour_map_simple(coords):
    """
    Generate a US map with state lines without cartopy.
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


def generate_tour_map_cartopy(coords):
    """
    Generate a high-quality US map using cartopy.
    Shows only continental US with state lines.
    """
    # Create figure with beige background
    fig = plt.figure(figsize=(10, 6), facecolor='#f9f5eb')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_facecolor('#f9f5eb')

    # Set extent FIRST to continental US bounds
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

    # Draw only US states (not global land mass) for cleaner look
    # Use STATES which only draws US state boundaries
    states = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lakes',
        scale='50m',
        facecolor='#e8e0d0',
        edgecolor='#aaaaaa'
    )
    ax.add_feature(states, linewidth=0.5, zorder=1)

    # Add coastline for cleaner edges
    ax.add_feature(cfeature.COASTLINE, edgecolor='#888888', linewidth=0.6, zorder=2)

    # Add Great Lakes
    ax.add_feature(cfeature.LAKES, facecolor='#d4e5e5', edgecolor='#888888', linewidth=0.3, zorder=2)

    # Plot show locations
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    # Draw glow effect first
    ax.scatter(lons, lats, c='#c9a227', s=350, zorder=4, alpha=0.3,
               transform=ccrs.PlateCarree())
    # Draw main dots
    ax.scatter(lons, lats, c='#c9a227', s=200, zorder=5,
               edgecolors='#1a1a1a', linewidths=2, alpha=0.95,
               transform=ccrs.PlateCarree())

    # Remove all axes, ticks, and frame
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150,
                facecolor='#f9f5eb', edgecolor='none', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue()


def generate_tour_map(shows):
    """
    Generate a US map PNG with dots for each show location.
    Uses caching to avoid regenerating the same map.
    """
    # Get unique locations
    unique_locations = sorted(set(show.get('location', '') for show in shows if show.get('location')))

    # Create cache key from locations
    cache_key = tuple(unique_locations)
    if cache_key in _tour_map_cache:
        print("Using cached tour map")
        return _tour_map_cache[cache_key]

    # Collect coordinates for unique locations (instant with lookup table)
    coords = []
    for location in unique_locations:
        result = geocode_location(location)
        if result:
            coords.append(result)

    if not coords:
        return None

    # Generate the map
    print(f"Generating tour map for {len(coords)} locations...")
    if CARTOPY_AVAILABLE:
        try:
            map_data = generate_tour_map_cartopy(coords)
        except Exception as e:
            print(f"Cartopy map failed, using fallback: {e}")
            map_data = generate_tour_map_simple(coords)
    else:
        map_data = generate_tour_map_simple(coords)

    # Cache the result
    _tour_map_cache[cache_key] = map_data
    return map_data


# Path to pre-generated HD tour map (committed to repo)
STATIC_FOLDER = Path(__file__).parent / 'static'
HD_MAP_PATH = STATIC_FOLDER / 'tour_map_hd.png'


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(STATIC_FOLDER, filename)


@app.route('/api/generate-hd-map', methods=['POST'])
def api_generate_hd_map():
    """
    Generate an HD tour map using cartopy and upload to Cloudinary.
    Falls back to local save if Cloudinary not configured.
    """
    if not CARTOPY_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'HD map generation requires cartopy. Run locally to generate and upload to Cloudinary.'
        })

    try:
        data = request.get_json()
        shows = data.get('shows', [])

        if not shows:
            return jsonify({'success': False, 'error': 'No shows provided'})

        # Collect coordinates
        coords = []
        for show in shows:
            location = show.get('location', '')
            if location:
                result = geocode_location(location)
                if result:
                    coords.append(result)

        if not coords:
            return jsonify({'success': False, 'error': 'Could not geocode any locations'})

        # Generate HD map with cartopy
        print(f"Generating HD tour map for {len(coords)} locations...")
        map_data = generate_tour_map_cartopy(coords)

        # Try to upload to Cloudinary
        if CLOUDINARY_AVAILABLE and CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY:
            try:
                print("Uploading tour map to Cloudinary...")
                result = cloudinary.uploader.upload(
                    map_data,
                    public_id="hoh_tour_map",
                    folder="newsletter",
                    overwrite=True,
                    resource_type="image"
                )
                cloudinary_url = result.get('secure_url')
                print(f"Uploaded to Cloudinary: {cloudinary_url}")
                return jsonify({
                    'success': True,
                    'message': f'Tour map uploaded to Cloudinary ({len(coords)} locations)',
                    'url': cloudinary_url,
                    'locations': len(coords)
                })
            except Exception as e:
                print(f"Cloudinary upload failed: {e}, falling back to local save")

        # Fallback: Save to static folder (for local dev)
        STATIC_FOLDER.mkdir(exist_ok=True)
        with open(HD_MAP_PATH, 'wb') as f:
            f.write(map_data)

        return jsonify({
            'success': True,
            'message': f'HD map saved locally ({len(coords)} locations). Commit and push to deploy to Render.',
            'path': str(HD_MAP_PATH),
            'locations': len(coords)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/tour-map', methods=['POST'])
def api_tour_map():
    """
    Get tour map image URL.
    Uses Cloudinary URL if configured, otherwise pre-generated HD map or dynamic generation.
    """
    try:
        data = request.get_json()
        shows = data.get('shows', [])

        if not shows:
            return jsonify({'success': False, 'error': 'No shows provided'})

        # Check if Cloudinary is configured - use the known URL
        if CLOUDINARY_CLOUD_NAME:
            cloudinary_url = f"https://res.cloudinary.com/{CLOUDINARY_CLOUD_NAME}/image/upload/newsletter/hoh_tour_map.png"
            print(f"Using Cloudinary tour map: {cloudinary_url}")
            return jsonify({'success': True, 'url': cloudinary_url, 'absolute_url': cloudinary_url, 'source': 'cloudinary'})

        # Check if HD map exists (pre-generated and committed)
        if HD_MAP_PATH.exists():
            print("Using pre-generated HD tour map")
            url = "/static/tour_map_hd.png"
            base_url = get_base_url()
            absolute_url = f"{base_url}{url}"
            return jsonify({'success': True, 'url': url, 'absolute_url': absolute_url, 'source': 'hd_static'})

        # Fall back to dynamic generation
        map_data = generate_tour_map(shows)

        if not map_data:
            return jsonify({'success': False, 'error': 'Could not generate map - no valid locations'})

        # Create a unique filename based on the show locations (so same shows = same file)
        locations_str = '|'.join(sorted(show.get('location', '') for show in shows))
        filename_hash = hashlib.md5(locations_str.encode()).hexdigest()[:12]
        filename = f"tourmap_{filename_hash}.png"
        filepath = UPLOAD_FOLDER / filename

        # Save the map image to uploads folder
        with open(filepath, 'wb') as f:
            f.write(map_data)

        # Return the URL (will be converted to absolute URL when building newsletter)
        url = f"/uploads/{filename}"

        # Get base URL for the full absolute URL
        base_url = get_base_url()
        absolute_url = f"{base_url}{url}"

        return jsonify({'success': True, 'url': url, 'absolute_url': absolute_url, 'source': 'dynamic'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    # Use PORT environment variable for cloud hosting (Render, etc.)
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'

    print("\n" + "="*50)
    print("  HOUSE OF HAMILL NEWSLETTER BUILDER")
    print(f"  Web UI running at http://localhost:{port}")
    print("="*50 + "\n")
    app.run(debug=debug, host='0.0.0.0', port=port)

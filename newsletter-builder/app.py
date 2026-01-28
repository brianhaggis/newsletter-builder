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
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from scrapers.merch import get_all_merch
from scrapers.shows import get_upcoming_shows
from config import COLORS, FONTS, DEFAULT_HEADER_IMAGE

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


def markdown_to_html(text):
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
        html_parts.append(f'<p style="margin: 0 0 16px 0;">{p}</p>')

    return '\n'.join(html_parts)


def build_newsletter_html(body_text, shows=None, merch=None, photo_url=None, subject="", tour_map_url=None):
    """
    Build the newsletter HTML from components.
    """
    # Set up Jinja
    template_dir = Path(__file__).parent / 'templates'
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('newsletter.html')

    # Check if body is already HTML (from rich text editor) or plain text
    if body_text and body_text.strip().startswith('<'):
        # Already HTML from Quill editor - add inline styles to paragraphs
        body_html = body_text.replace('<p>', '<p style="margin: 0 0 16px 0;">')
    else:
        # Convert plain text to HTML
        body_html = markdown_to_html(body_text)

    # Render template
    html = template.render(
        subject=subject,
        body_html=body_html,
        shows=shows or [],
        merch=merch,
        photo_url=photo_url,
        tour_map_url=tour_map_url,
        year=datetime.now().year,
        colors=COLORS,
        fonts=FONTS,
    )

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


@app.route('/api/preview', methods=['POST'])
def api_preview():
    """Generate preview HTML."""
    data = request.get_json()

    subject = data.get('subject', '')
    body_text = data.get('body', '')
    photo_url = data.get('photo_url') or None
    merch = data.get('merch') or None
    shows = data.get('shows') or []
    tour_map_url = data.get('tour_map_url') or None

    html = build_newsletter_html(
        body_text=body_text,
        shows=shows,
        merch=merch,
        photo_url=photo_url,
        subject=subject,
        tour_map_url=tour_map_url
    )

    return jsonify({'success': True, 'html': html})


@app.route('/api/download', methods=['POST'])
def api_download():
    """Download HTML file."""
    data = request.get_json()

    subject = data.get('subject', '')
    body_text = data.get('body', '')
    photo_url = data.get('photo_url') or None
    merch = data.get('merch') or None
    shows = data.get('shows') or []
    tour_map_url = data.get('tour_map_url') or None

    html = build_newsletter_html(
        body_text=body_text,
        shows=shows,
        merch=merch,
        photo_url=photo_url,
        subject=subject,
        tour_map_url=tour_map_url
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
    """Clear cache and refresh data."""
    global _cache
    _cache = {}
    return jsonify({'success': True, 'message': 'Cache cleared'})


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


# Geocoding cache to avoid repeated API calls
_geocode_cache = {}

def geocode_location(location_str):
    """
    Convert a location string to lat/lon coordinates.
    Uses Nominatim with caching.
    """
    if location_str in _geocode_cache:
        return _geocode_cache[location_str]

    try:
        geolocator = Nominatim(user_agent="newsletter-builder")
        location = geolocator.geocode(location_str, timeout=5)
        if location:
            coords = (location.latitude, location.longitude)
            _geocode_cache[location_str] = coords
            return coords
    except (GeocoderTimedOut, Exception) as e:
        print(f"Geocoding error for {location_str}: {e}")

    _geocode_cache[location_str] = None
    return None


def generate_tour_map(shows):
    """
    Generate a US map PNG with dots for each show location.
    Returns base64-encoded PNG data.
    """
    # Collect coordinates for all shows
    coords = []
    for show in shows:
        location = show.get('location', '')
        if location:
            result = geocode_location(location)
            if result:
                coords.append(result)
            time.sleep(0.1)  # Rate limit geocoding

    if not coords:
        return None

    # Create the map figure
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f9f5eb')
    ax.set_facecolor('#f9f5eb')

    # US boundaries (approximate continental US)
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)

    # Draw a simple US outline (approximate)
    # Using a simplified polygon for the continental US
    us_outline_x = [-124, -117, -114, -111, -104, -104, -100, -100, -97, -97,
                   -94, -94, -90, -89, -82, -81, -81, -75, -75, -70, -67, -67,
                   -70, -71, -70, -67, -67, -80, -80, -82, -85, -88, -89, -89,
                   -94, -97, -97, -100, -103, -109, -111, -114, -117, -120, -124, -124]
    us_outline_y = [42, 42, 35, 35, 35, 37, 37, 40, 40, 37, 37, 33, 33, 30, 30,
                   25, 31, 31, 35, 35, 35, 41, 41, 42, 44, 45, 47, 47, 45, 45,
                   47, 47, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 42]

    # Draw state-like background
    ax.fill(us_outline_x, us_outline_y, color='#e8e0d0', alpha=0.5, linewidth=2, edgecolor='#ccc')

    # Plot show locations
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    # Draw dots with gold color matching the brand
    ax.scatter(lons, lats, c='#c9a227', s=120, zorder=5, edgecolors='#1a1a1a', linewidths=1.5, alpha=0.9)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='#f9f5eb')
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue()


@app.route('/api/tour-map', methods=['POST'])
def api_tour_map():
    """
    Generate a tour map image.
    Accepts list of shows in request body.
    Returns base64 PNG or URL to saved image.
    """
    try:
        data = request.get_json()
        shows = data.get('shows', [])

        if not shows:
            return jsonify({'success': False, 'error': 'No shows provided'})

        # Generate the map
        map_data = generate_tour_map(shows)

        if not map_data:
            return jsonify({'success': False, 'error': 'Could not generate map - no valid locations'})

        # Save to uploads folder
        filename = f"tour_map_{uuid.uuid4().hex[:8]}.png"
        filepath = UPLOAD_FOLDER / filename

        with open(filepath, 'wb') as f:
            f.write(map_data)

        url = f"/uploads/{filename}"
        return jsonify({'success': True, 'url': url})

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

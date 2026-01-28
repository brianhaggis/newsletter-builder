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
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

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


def generate_tour_map_simple(coords):
    """
    Generate a simple US map without cartopy.
    Uses a stylized outline of the continental US.
    """
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#f9f5eb')
    ax.set_facecolor('#f9f5eb')

    # Set extent to continental US
    ax.set_xlim(-128, -64)
    ax.set_ylim(22, 52)

    # Simplified US continental outline (more detailed than before)
    us_outline = [
        (-124.7, 48.4), (-124.2, 46.2), (-123.5, 46.2), (-124.1, 43.7),
        (-124.4, 40.3), (-122.4, 37.2), (-121.5, 36.0), (-117.1, 32.5),
        (-114.7, 32.7), (-111.1, 31.3), (-108.2, 31.3), (-106.5, 31.8),
        (-104.0, 29.5), (-103.0, 29.0), (-102.1, 29.8), (-99.5, 27.5),
        (-97.4, 26.0), (-97.1, 28.0), (-94.0, 29.5), (-90.0, 29.0),
        (-89.0, 29.2), (-88.8, 30.4), (-86.8, 30.4), (-85.0, 29.0),
        (-83.0, 29.0), (-82.5, 27.5), (-80.0, 25.0), (-80.0, 26.5),
        (-81.5, 30.8), (-80.5, 32.0), (-78.5, 34.0), (-75.5, 35.2),
        (-75.5, 37.0), (-76.0, 38.0), (-75.2, 39.5), (-74.0, 39.5),
        (-74.0, 41.0), (-71.0, 41.3), (-70.0, 42.0), (-70.7, 43.0),
        (-68.0, 44.4), (-67.0, 45.0), (-67.0, 47.5), (-69.0, 47.5),
        (-70.0, 46.0), (-71.0, 45.0), (-75.0, 45.0), (-79.0, 43.5),
        (-82.5, 46.0), (-84.5, 46.5), (-88.0, 48.0), (-89.5, 48.0),
        (-95.0, 49.0), (-123.0, 49.0), (-124.7, 48.4)
    ]

    # Draw US outline
    xs, ys = zip(*us_outline)
    ax.fill(xs, ys, color='#e8e0d0', edgecolor='#999999', linewidth=1.5, zorder=1)

    # Plot show locations
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    # Draw glow
    ax.scatter(lons, lats, c='#c9a227', s=400, zorder=4, alpha=0.3)
    # Draw main dots
    ax.scatter(lons, lats, c='#c9a227', s=180, zorder=5,
               edgecolors='#1a1a1a', linewidths=2, alpha=0.95)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200,
                facecolor='#f9f5eb', edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue()


def generate_tour_map_cartopy(coords):
    """
    Generate a high-quality US map using cartopy.
    """
    fig = plt.figure(figsize=(12, 7), facecolor='#f9f5eb')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
        central_longitude=-96, central_latitude=39
    ))

    # Set extent to continental US
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='#e8e0d0')
    ax.add_feature(cfeature.OCEAN, facecolor='#d4e5e5')
    ax.add_feature(cfeature.LAKES, facecolor='#d4e5e5', alpha=0.5)
    ax.add_feature(cfeature.STATES, edgecolor='#999999', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor='#666666', linewidth=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='#666666', linewidth=0.8)

    # Plot show locations
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    # Draw dots with gold color matching the brand - with glow effect
    ax.scatter(lons, lats, c='#c9a227', s=200, zorder=6,
               edgecolors='#1a1a1a', linewidths=2, alpha=0.95,
               transform=ccrs.PlateCarree())
    # Add subtle glow
    ax.scatter(lons, lats, c='#c9a227', s=350, zorder=5,
               alpha=0.3, transform=ccrs.PlateCarree())

    # Remove frame
    ax.spines['geo'].set_visible(False)

    # Save to bytes at high resolution
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200,
                facecolor='#f9f5eb', edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue()


def generate_tour_map(shows):
    """
    Generate a US map PNG with dots for each show location.
    Uses cartopy if available, otherwise falls back to simple map.
    """
    # Get unique locations to minimize geocoding calls
    unique_locations = list(set(show.get('location', '') for show in shows if show.get('location')))

    # Collect coordinates for unique locations
    coords = []
    for location in unique_locations:
        # Check cache first (no delay needed for cached results)
        if location in _geocode_cache:
            result = _geocode_cache[location]
            if result:
                coords.append(result)
            continue

        # Geocode new location with rate limiting
        result = geocode_location(location)
        if result:
            coords.append(result)
        time.sleep(0.05)  # Reduced delay - only for uncached requests

    if not coords:
        return None

    # Use cartopy if available, otherwise use simple fallback
    if CARTOPY_AVAILABLE:
        try:
            return generate_tour_map_cartopy(coords)
        except Exception as e:
            print(f"Cartopy map failed, using fallback: {e}")
            return generate_tour_map_simple(coords)
    else:
        return generate_tour_map_simple(coords)


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

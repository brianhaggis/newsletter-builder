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

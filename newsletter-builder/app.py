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
from pathlib import Path
from datetime import datetime
from functools import wraps
import time

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from jinja2 import Environment, FileSystemLoader
from PIL import Image

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


def build_newsletter_html(body_text, shows=None, merch=None, photo_url=None, subject=""):
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

    html = build_newsletter_html(
        body_text=body_text,
        shows=shows,
        merch=merch,
        photo_url=photo_url,
        subject=subject
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

    html = build_newsletter_html(
        body_text=body_text,
        shows=shows,
        merch=merch,
        photo_url=photo_url,
        subject=subject
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


if __name__ == '__main__':
    # Use PORT environment variable for cloud hosting (Render, etc.)
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'

    print("\n" + "="*50)
    print("  HOUSE OF HAMILL NEWSLETTER BUILDER")
    print(f"  Web UI running at http://localhost:{port}")
    print("="*50 + "\n")
    app.run(debug=debug, host='0.0.0.0', port=port)

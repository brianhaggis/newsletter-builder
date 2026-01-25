#!/usr/bin/env python3
"""
House of Hamill Newsletter Builder

Pulls tour dates from Bandsintown, scrapes merch from the website,
and generates email-safe HTML newsletters.

Usage:
    python builder.py              # Interactive mode
    python builder.py --preview    # Quick preview with sample content
"""

import argparse
import os
import sys
import webbrowser
import tempfile
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from jinja2 import Environment, FileSystemLoader
import pyperclip

from scrapers.merch import get_all_merch, get_in_stock_merch, get_random_spotlight, get_spotlight_by_name
from scrapers.shows import get_upcoming_shows, format_shows_for_display
from config import COLORS, FONTS


def markdown_to_html(text):
    """
    Convert simple markdown-ish text to HTML paragraphs.
    Handles line breaks and basic formatting.
    """
    # Split into paragraphs (double newline)
    paragraphs = text.strip().split('\n\n')
    
    html_parts = []
    for p in paragraphs:
        # Replace single newlines with <br>
        p = p.replace('\n', '<br>')
        html_parts.append(f'<p style="margin: 0 0 16px 0;">{p}</p>')
    
    return '\n'.join(html_parts)


def get_body_text():
    """
    Get the newsletter body text from the user.
    Opens a temp file in their default editor.
    """
    # Create a temp file with instructions
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("# Write your newsletter content below this line.\n")
        f.write("# Blank lines create new paragraphs.\n")
        f.write("# Save and close the file when done.\n")
        f.write("# Lines starting with # will be removed.\n\n")
        temp_path = f.name
    
    # Try to open in default editor
    editor = os.environ.get('EDITOR', 'nano')
    
    print(f"\nOpening editor ({editor})...")
    print("Write your newsletter, save, and close the editor.\n")
    
    os.system(f'{editor} "{temp_path}"')
    
    # Read the content
    with open(temp_path, 'r') as f:
        content = f.read()
    
    # Clean up temp file
    os.unlink(temp_path)
    
    # Remove comment lines
    lines = [l for l in content.split('\n') if not l.strip().startswith('#')]
    return '\n'.join(lines).strip()


def get_body_text_interactive():
    """
    Get body text via terminal input (fallback if editor doesn't work).
    """
    print("\nEnter your newsletter text (press Enter twice when done):\n")
    lines = []
    empty_count = 0
    
    while True:
        line = input()
        if line == '':
            empty_count += 1
            if empty_count >= 2:
                break
            lines.append('')
        else:
            empty_count = 0
            lines.append(line)
    
    return '\n'.join(lines).strip()


def choose_merch_spotlight(products):
    """Let user choose a merch item or pick random."""
    print("\n--- MERCH SPOTLIGHT ---")
    print("Available in-stock items:\n")
    
    in_stock = [p for p in products if p['in_stock']]
    
    for i, p in enumerate(in_stock, 1):
        sale = " (SALE!)" if p['on_sale'] else ""
        print(f"  {i}. {p['name']} - ${p['price']}{sale}")
    
    print(f"\n  R. Random pick")
    print(f"  S. Skip merch spotlight")
    
    choice = input("\nYour choice: ").strip().upper()
    
    if choice == 'S':
        return None
    elif choice == 'R':
        import random
        return random.choice(in_stock)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(in_stock):
                return in_stock[idx]
        except ValueError:
            pass
    
    # Default to random
    import random
    return random.choice(in_stock)


def build_newsletter(body_text, shows=None, merch=None, photo_url=None, subject=""):
    """
    Build the newsletter HTML from components.
    """
    # Set up Jinja
    template_dir = Path(__file__).parent / 'templates'
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('newsletter.html')
    
    # Convert body text to HTML
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


def save_and_preview(html, filename=None):
    """
    Save the newsletter and open preview in browser.
    Also copy to clipboard.
    """
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"newsletter_{timestamp}.html"
    
    # Save to output directory
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"\n✓ Saved to: {output_path}")
    
    # Copy to clipboard
    try:
        pyperclip.copy(html)
        print("✓ Copied to clipboard")
    except Exception as e:
        print(f"  (Clipboard copy failed: {e})")
    
    # Open in browser
    print("✓ Opening preview in browser...")
    webbrowser.open(f'file://{output_path.absolute()}')
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='House of Hamill Newsletter Builder')
    parser.add_argument('--preview', action='store_true', help='Quick preview with sample content')
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("  HOUSE OF HAMILL NEWSLETTER BUILDER")
    print("="*50)
    
    # Fetch tour dates
    print("\n[1/3] Fetching tour dates...")
    shows = get_upcoming_shows()
    if shows:
        print(f"  Found {len(shows)} upcoming shows")
    else:
        print("  No shows found (check API key in config.py)")
    
    # Fetch merch
    print("\n[2/3] Scraping merch page...")
    try:
        products = get_all_merch()
        in_stock = [p for p in products if p['in_stock']]
        print(f"  Found {len(products)} products ({len(in_stock)} in stock)")
    except Exception as e:
        print(f"  Error fetching merch: {e}")
        products = []
        in_stock = []
    
    # Quick preview mode
    if args.preview:
        print("\n[3/3] Building preview...")
        body_text = "This is a preview newsletter with sample content.\n\nYour actual newsletter text will go here. Write about upcoming shows, new music, or whatever's on your mind."
        merch = in_stock[0] if in_stock else None
        html = build_newsletter(body_text, shows, merch, subject="Preview Newsletter")
        save_and_preview(html, "preview.html")
        return
    
    # Interactive mode
    print("\n[3/3] Building newsletter...")
    
    # Get subject line
    subject = input("\nSubject line: ").strip()
    
    # Get photo URL (optional)
    photo_url = input("Featured photo URL (or press Enter to skip): ").strip() or None
    
    # Choose merch spotlight
    merch = None
    if in_stock:
        merch = choose_merch_spotlight(products)
        if merch:
            print(f"\n  Selected: {merch['name']}")
    
    # Get body text
    print("\n--- NEWSLETTER BODY ---")
    use_editor = input("Open text editor? (Y/n): ").strip().lower() != 'n'
    
    if use_editor:
        body_text = get_body_text()
    else:
        body_text = get_body_text_interactive()
    
    if not body_text:
        print("\nNo content entered. Exiting.")
        return
    
    # Build and save
    html = build_newsletter(body_text, shows, merch, photo_url, subject)
    save_and_preview(html)
    
    print("\n" + "="*50)
    print("  Done! Paste the HTML into Bandzoogle.")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()

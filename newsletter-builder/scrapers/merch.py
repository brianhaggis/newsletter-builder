"""
Scraper for merchandise pages.
Extracts product names, descriptions, prices, images, and stock status.
Supports multiple bands.
"""

import requests
from bs4 import BeautifulSoup
import random
import re
from config import MERCH_URL, BANDS, DEFAULT_BAND


def fetch_merch_page(merch_url=None):
    """Fetch the raw HTML from the merch page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    url = merch_url or MERCH_URL
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def parse_products(html):
    """
    Parse products from the merch page HTML.
    Returns a list of dicts with product info.
    """
    soup = BeautifulSoup(html, "html.parser")
    products = []

    # Find all product containers - they use h1 or h2 tags for product names
    # and are wrapped in recognizable structures
    product_titles = soup.find_all(["h1", "h2"])

    for title_tag in product_titles:
        # Skip navigation/header h1s
        title_text = title_tag.get_text(strip=True)
        if not title_text or "house of hamill" in title_text.lower() and "upcycled" in title_text.lower():
            continue
        if title_text in ["MERCHANDISE", "Frequently purchased together"]:
            continue

        product = {
            "name": title_text,
            "description": "",
            "price": None,
            "image_url": None,
            "in_stock": True,
            "on_sale": False,
            "sale_price": None,
        }

        # Look for description - find the product container and look for <p> tags
        container = title_tag
        for _ in range(5):  # Go up to find the product container
            container = container.find_parent() if container else None
            if container and container.name == 'div':
                # Look for paragraph text in this container
                paragraphs = container.find_all('p', recursive=True)
                desc_parts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    # Skip price, cart buttons, stock status, size selectors
                    if text and not text.startswith("$") and "Add to cart" not in text:
                        if "Out of stock" not in text and "Size" not in text and "Color" not in text:
                            if len(text) > 20:  # Only substantial text
                                desc_parts.append(text)
                if desc_parts:
                    product["description"] = desc_parts[0][:200]  # First description, truncated
                    break

        # Look for price - find text containing $ near this product
        # Go up multiple parent levels to find the product container with price
        container = title_tag
        for _ in range(3):
            container = container.find_parent() if container else None
        if container:
            price_text = container.get_text()
            # Look for sale price pattern
            sale_match = re.search(r'Sale\$?([\d.]+)\$?([\d.]+)', price_text)
            if sale_match:
                product["on_sale"] = True
                product["price"] = float(sale_match.group(1))
                product["sale_price"] = float(sale_match.group(2))
            else:
                # Regular price
                price_match = re.search(r'\$([\d.]+)', price_text)
                if price_match:
                    product["price"] = float(price_match.group(1))

            # Check stock status
            if "Out of stock" in price_text or "Not available" in price_text:
                # Check if ALL variants are out of stock
                if price_text.count("Out of stock") > 3:  # Multiple sizes out
                    product["in_stock"] = False

        # Look for image - find the product image associated with this item
        img_url = None

        # Strategy 1: Look for the closest previous <a> containing an <img>
        # This works well for most Bandzoogle/store layouts
        prev_link = title_tag.find_previous("a")
        if prev_link:
            img = prev_link.find("img")
            if img:
                img_url = img.get("src") or img.get("data-src")

        # Strategy 2: If no image in previous link, check previous siblings
        if not img_url:
            prev_sibling = title_tag.find_previous_sibling()
            while prev_sibling and not img_url:
                if prev_sibling.name == "img":
                    img_url = prev_sibling.get("src") or prev_sibling.get("data-src")
                elif prev_sibling.name in ["a", "div", "figure"]:
                    img = prev_sibling.find("img")
                    if img:
                        img_url = img.get("src") or img.get("data-src")
                prev_sibling = prev_sibling.find_previous_sibling() if not img_url else None

        # Strategy 3: Check immediate parent for images before the title
        if not img_url:
            parent = title_tag.find_parent()
            if parent:
                # Get all images in parent that appear before title
                for elem in parent.children:
                    if elem == title_tag:
                        break
                    if hasattr(elem, 'find'):
                        if elem.name == "img":
                            img_url = elem.get("src") or elem.get("data-src")
                        else:
                            img = elem.find("img") if hasattr(elem, 'find') else None
                            if img:
                                img_url = img.get("src") or img.get("data-src")
                    if img_url:
                        break

        if img_url:
            # Clean up the URL - get larger version
            if "resize" in img_url:
                img_url = re.sub(r'resize.*?\]', 'resize",600]', img_url)
            if img_url.startswith("//"):
                img_url = "https:" + img_url
            product["image_url"] = img_url

        # Only add products that have a name and price
        if product["name"] and product["price"]:
            products.append(product)

    return products


def get_all_merch(band_id=None):
    """
    Get all merchandise from the store.

    Args:
        band_id: Band identifier (e.g., 'house_of_hamill', 'enter_the_haggis')
                 If None, uses legacy MERCH_URL for backwards compatibility.
    """
    # Get merch URL from band config or fall back to legacy
    if band_id and band_id in BANDS:
        merch_url = BANDS[band_id]["merch_url"]
    else:
        merch_url = MERCH_URL  # Legacy fallback

    html = fetch_merch_page(merch_url)
    return parse_products(html)


def get_in_stock_merch():
    """Get only in-stock merchandise."""
    all_merch = get_all_merch()
    return [p for p in all_merch if p["in_stock"]]


def get_random_spotlight():
    """Pick a random in-stock item for the newsletter spotlight."""
    in_stock = get_in_stock_merch()
    if not in_stock:
        return None
    return random.choice(in_stock)


def get_spotlight_by_name(name_fragment):
    """Find a specific product by partial name match."""
    all_merch = get_all_merch()
    name_lower = name_fragment.lower()
    matches = [p for p in all_merch if name_lower in p["name"].lower()]
    return matches[0] if matches else None


if __name__ == "__main__":
    # Test the scraper
    print("Fetching merch...")
    products = get_all_merch()
    print(f"\nFound {len(products)} products:\n")
    for p in products:
        stock = "✓" if p["in_stock"] else "✗"
        sale = " (SALE!)" if p["on_sale"] else ""
        print(f"[{stock}] {p['name']} - ${p['price']}{sale}")
        if p["description"]:
            print(f"    {p['description'][:80]}...")
        print()

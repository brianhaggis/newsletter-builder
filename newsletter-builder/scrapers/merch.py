"""
Scraper for House of Hamill merchandise page.
Extracts product names, descriptions, prices, images, and stock status.
"""

import requests
from bs4 import BeautifulSoup
import random
import re
from config import MERCH_URL


def fetch_merch_page():
    """Fetch the raw HTML from the merch page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    response = requests.get(MERCH_URL, headers=headers)
    response.raise_for_status()
    return response.text


def parse_products(html):
    """
    Parse products from the merch page HTML.
    Returns a list of dicts with product info.
    """
    soup = BeautifulSoup(html, "html.parser")
    products = []

    # Find all product containers - they use h1 tags for product names
    # and are wrapped in recognizable structures
    product_titles = soup.find_all("h1")

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

        # Look for description - usually in a paragraph or text near the title
        parent = title_tag.find_parent()
        if parent:
            # Find description text
            desc_parts = []
            for sibling in title_tag.find_next_siblings():
                if sibling.name == "h1":
                    break
                text = sibling.get_text(strip=True)
                if text and not text.startswith("$") and "Add to cart" not in text:
                    # Skip size/variant tables
                    if "Size" not in text and "Color" not in text:
                        desc_parts.append(text)
            if desc_parts:
                product["description"] = " ".join(desc_parts[:2])  # First couple chunks

        # Look for price - find text containing $ near this product
        container = title_tag.find_parent("div") or title_tag.find_parent()
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

        # Look for image - find nearby img tag
        # Images are usually in a link before or near the title
        img_container = title_tag.find_previous("a")
        if img_container:
            img = img_container.find("img")
            if img and img.get("src"):
                img_url = img["src"]
                # Clean up the URL - get larger version
                if "resize" in img_url:
                    # Try to get a larger image by modifying the URL
                    img_url = re.sub(r'resize.*?\]', 'resize",600]', img_url)
                if img_url.startswith("//"):
                    img_url = "https:" + img_url
                product["image_url"] = img_url

        # Only add products that have a name and price
        if product["name"] and product["price"]:
            products.append(product)

    return products


def get_all_merch():
    """Get all merchandise from the store."""
    html = fetch_merch_page()
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

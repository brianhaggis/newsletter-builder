#!/usr/bin/env python3
"""
House of Hamill Tour Map Updater
Double-click to fetch shows, generate map, and push to Render.
"""

import os
import sys
import io
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'newsletter-builder'))

import requests
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try to import cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("ERROR: Cartopy not installed. Install with: pip3 install cartopy")
    sys.exit(1)

from geopy.geocoders import Nominatim

# Config
BANDSINTOWN_API_KEY = "bd6845ee17cbcb49d93e3de843d13a21"
ARTIST_NAME = "House of Hamill"
STATIC_FOLDER = PROJECT_ROOT / 'newsletter-builder' / 'static'
HD_MAP_PATH = STATIC_FOLDER / 'tour_map_hd.png'

# City coordinates lookup
US_CITY_COORDS = {
    "birmingham, al": (33.5207, -86.8025), "huntsville, al": (34.7304, -86.5861),
    "mobile, al": (30.6954, -88.0399), "montgomery, al": (32.3792, -86.3077),
    "phoenix, az": (33.4484, -112.0740), "tucson, az": (32.2226, -110.9747),
    "scottsdale, az": (33.4942, -111.9261), "tempe, az": (33.4255, -111.9400),
    "little rock, ar": (34.7465, -92.2896), "fayetteville, ar": (36.0626, -94.1574),
    "los angeles, ca": (34.0522, -118.2437), "san francisco, ca": (37.7749, -122.4194),
    "san diego, ca": (32.7157, -117.1611), "san jose, ca": (37.3382, -121.8863),
    "oakland, ca": (37.8044, -122.2712), "sacramento, ca": (38.5816, -121.4944),
    "santa barbara, ca": (34.4208, -119.6982), "berkeley, ca": (37.8716, -122.2727),
    "denver, co": (39.7392, -104.9903), "boulder, co": (40.0150, -105.2705),
    "colorado springs, co": (38.8339, -104.8214), "fort collins, co": (40.5853, -105.0844),
    "hartford, ct": (41.7658, -72.6734), "new haven, ct": (41.3083, -72.9279),
    "wilmington, de": (39.7391, -75.5398), "dover, de": (39.1582, -75.5244),
    "miami, fl": (25.7617, -80.1918), "orlando, fl": (28.5383, -81.3792),
    "tampa, fl": (27.9506, -82.4572), "jacksonville, fl": (30.3322, -81.6557),
    "atlanta, ga": (33.7490, -84.3880), "savannah, ga": (32.0809, -81.0912),
    "athens, ga": (33.9519, -83.3576), "augusta, ga": (33.4735, -82.0105),
    "boise, id": (43.6150, -116.2023),
    "chicago, il": (41.8781, -87.6298), "springfield, il": (39.7817, -89.6501),
    "indianapolis, in": (39.7684, -86.1581), "bloomington, in": (39.1653, -86.5264),
    "des moines, ia": (41.5868, -93.6250), "iowa city, ia": (41.6611, -91.5302),
    "kansas city, ks": (39.1141, -94.6275), "wichita, ks": (37.6872, -97.3301),
    "louisville, ky": (38.2527, -85.7585), "lexington, ky": (38.0406, -84.5037),
    "new orleans, la": (29.9511, -90.0715), "baton rouge, la": (30.4515, -91.1871),
    "portland, me": (43.6591, -70.2568), "bangor, me": (44.8016, -68.7712),
    "baltimore, md": (39.2904, -76.6122), "annapolis, md": (38.9784, -76.4922),
    "boston, ma": (42.3601, -71.0589), "cambridge, ma": (42.3736, -71.1097),
    "worcester, ma": (42.2626, -71.8023), "northampton, ma": (42.3251, -72.6412),
    "detroit, mi": (42.3314, -83.0458), "ann arbor, mi": (42.2808, -83.7430),
    "grand rapids, mi": (42.9634, -85.6681), "lansing, mi": (42.7325, -84.5555),
    "minneapolis, mn": (44.9778, -93.2650), "st. paul, mn": (44.9537, -93.0900),
    "jackson, ms": (32.2988, -90.1848), "oxford, ms": (34.3665, -89.5192),
    "st. louis, mo": (38.6270, -90.1994), "kansas city, mo": (39.0997, -94.5786),
    "missoula, mt": (46.8721, -113.9940), "bozeman, mt": (45.6770, -111.0429),
    "omaha, ne": (41.2565, -95.9345), "lincoln, ne": (40.8258, -96.6852),
    "las vegas, nv": (36.1699, -115.1398), "reno, nv": (39.5296, -119.8138),
    "manchester, nh": (42.9956, -71.4548), "portsmouth, nh": (43.0718, -70.7626),
    "newark, nj": (40.7357, -74.1724), "jersey city, nj": (40.7178, -74.0431),
    "albuquerque, nm": (35.0844, -106.6504), "santa fe, nm": (35.6870, -105.9378),
    "new york, ny": (40.7128, -74.0060), "brooklyn, ny": (40.6782, -73.9442),
    "buffalo, ny": (42.8864, -78.8784), "albany, ny": (42.6526, -73.7562),
    "rochester, ny": (43.1566, -77.6088), "syracuse, ny": (43.0481, -76.1474),
    "charlotte, nc": (35.2271, -80.8431), "raleigh, nc": (35.7796, -78.6382),
    "durham, nc": (35.9940, -78.8986), "asheville, nc": (35.5951, -82.5515),
    "fargo, nd": (46.8772, -96.7898),
    "columbus, oh": (39.9612, -82.9988), "cleveland, oh": (41.4993, -81.6944),
    "cincinnati, oh": (39.1031, -84.5120), "dayton, oh": (39.7589, -84.1916),
    "oklahoma city, ok": (35.4676, -97.5164), "tulsa, ok": (36.1540, -95.9928),
    "portland, or": (45.5152, -122.6784), "eugene, or": (44.0521, -123.0868),
    "philadelphia, pa": (39.9526, -75.1652), "pittsburgh, pa": (40.4406, -79.9959),
    "state college, pa": (40.7934, -77.8600), "harrisburg, pa": (40.2732, -76.8867),
    "lancaster, pa": (40.0379, -76.3055),
    "providence, ri": (41.8240, -71.4128), "newport, ri": (41.4901, -71.3128),
    "charleston, sc": (32.7765, -79.9311), "columbia, sc": (34.0007, -81.0348),
    "sioux falls, sd": (43.5446, -96.7311),
    "nashville, tn": (36.1627, -86.7816), "memphis, tn": (35.1495, -90.0490),
    "knoxville, tn": (35.9606, -83.9207), "chattanooga, tn": (35.0456, -85.3097),
    "houston, tx": (29.7604, -95.3698), "austin, tx": (30.2672, -97.7431),
    "dallas, tx": (32.7767, -96.7970), "san antonio, tx": (29.4241, -98.4936),
    "salt lake city, ut": (40.7608, -111.8910), "provo, ut": (40.2338, -111.6585),
    "burlington, vt": (44.4759, -73.2121), "montpelier, vt": (44.2601, -72.5754),
    "richmond, va": (37.5407, -77.4360), "norfolk, va": (36.8508, -76.2859),
    "charlottesville, va": (38.0293, -78.4767), "virginia beach, va": (36.8529, -75.9780),
    "seattle, wa": (47.6062, -122.3321), "spokane, wa": (47.6588, -117.4260),
    "washington, dc": (38.9072, -77.0369),
    "charleston, wv": (38.3498, -81.6326), "morgantown, wv": (39.6295, -79.9559),
    "milwaukee, wi": (43.0389, -87.9065), "madison, wi": (43.0731, -89.4012),
    "cheyenne, wy": (41.1400, -104.8202), "jackson, wy": (43.4799, -110.7624),
}

_geocode_cache = {}


def get_upcoming_shows():
    """Fetch upcoming shows from Bandsintown API."""
    artist_encoded = requests.utils.quote(ARTIST_NAME)
    url = f"https://rest.bandsintown.com/artists/{artist_encoded}/events"
    params = {"app_id": BANDSINTOWN_API_KEY}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        events = response.json()
    except Exception as e:
        print(f"Error fetching shows: {e}")
        return []

    shows = []
    for event in events:
        venue = event.get("venue", {})
        city = venue.get("city", "")
        region = venue.get("region", "")
        country = venue.get("country", "")

        if country in ("United States", "US"):
            location = f"{city}, {region}" if region else city
        else:
            location = f"{city}, {country}" if country else city

        shows.append({
            "venue": venue.get("name", "TBA"),
            "location": location,
        })

    return shows


def geocode_location(location_str):
    """Convert location string to coordinates."""
    if not location_str:
        return None

    loc_lower = location_str.lower().strip()

    # Check lookup table
    if loc_lower in US_CITY_COORDS:
        return US_CITY_COORDS[loc_lower]

    # Try partial matching
    for key, coords in US_CITY_COORDS.items():
        if key.split(',')[0] in loc_lower or loc_lower.split(',')[0].strip() in key:
            return coords

    # Fall back to geocoding API
    if location_str in _geocode_cache:
        return _geocode_cache[location_str]

    try:
        geolocator = Nominatim(user_agent="newsletter-builder-hoh")
        location = geolocator.geocode(location_str, timeout=2)
        if location:
            coords = (location.latitude, location.longitude)
            _geocode_cache[location_str] = coords
            return coords
    except Exception:
        pass

    return None


def generate_tour_map(coords):
    """Generate HD tour map with cartopy."""
    fig = plt.figure(figsize=(10, 6), facecolor='#f9f5eb')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_facecolor('#f9f5eb')
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

    states = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lakes',
        scale='50m',
        facecolor='#e8e0d0',
        edgecolor='#aaaaaa'
    )
    ax.add_feature(states, linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='#888888', linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.LAKES, facecolor='#d4e5e5', edgecolor='#888888', linewidth=0.3, zorder=2)

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    ax.scatter(lons, lats, c='#c9a227', s=350, zorder=4, alpha=0.3, transform=ccrs.PlateCarree())
    ax.scatter(lons, lats, c='#c9a227', s=200, zorder=5, edgecolors='#1a1a1a', linewidths=2, alpha=0.95, transform=ccrs.PlateCarree())

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='#f9f5eb', edgecolor='none', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue()


def main():
    print("=" * 50)
    print("  HOUSE OF HAMILL TOUR MAP UPDATER")
    print("=" * 50)
    print()

    # Fetch shows
    print("Fetching shows from Bandsintown...")
    shows = get_upcoming_shows()
    if not shows:
        print("No shows found!")
        input("\nPress Enter to close...")
        return

    print(f"Found {len(shows)} shows")

    # Geocode locations
    print("\nGeocoding locations...")
    coords = []
    for show in shows:
        location = show.get('location', '')
        if location:
            result = geocode_location(location)
            if result:
                coords.append(result)
                print(f"  {location}")

    if not coords:
        print("Could not geocode any locations!")
        input("\nPress Enter to close...")
        return

    print(f"\nMapped {len(coords)} locations")

    # Generate map
    print("\nGenerating HD tour map...")
    map_data = generate_tour_map(coords)

    # Save map
    STATIC_FOLDER.mkdir(exist_ok=True)
    with open(HD_MAP_PATH, 'wb') as f:
        f.write(map_data)
    print(f"Saved to {HD_MAP_PATH}")

    # Git operations
    print("\nPushing to GitHub...")
    try:
        subprocess.run(
            ['git', 'add', 'newsletter-builder/static/tour_map_hd.png'],
            cwd=PROJECT_ROOT, check=True, capture_output=True
        )
        subprocess.run(
            ['git', 'commit', '-m', f'Update tour map ({len(coords)} locations)'],
            cwd=PROJECT_ROOT, check=True, capture_output=True
        )
        subprocess.run(
            ['git', 'push', 'origin', 'main'],
            cwd=PROJECT_ROOT, check=True, capture_output=True
        )
        print("Pushed to GitHub!")
        print("\nRender will auto-deploy shortly.")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        if "nothing to commit" in error_msg:
            print("Map unchanged - nothing to push.")
        else:
            print(f"Git error: {error_msg}")

    print("\n" + "=" * 50)
    print("  DONE!")
    print("=" * 50)
    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()

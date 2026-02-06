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

# City coordinates lookup - expanded to cover common tour venues
US_CITY_COORDS = {
    # Alabama
    "birmingham, al": (33.5207, -86.8025), "huntsville, al": (34.7304, -86.5861),
    "mobile, al": (30.6954, -88.0399), "montgomery, al": (32.3792, -86.3077),
    "auburn, al": (32.6099, -85.4808), "tuscaloosa, al": (33.2098, -87.5692),
    # Arizona
    "phoenix, az": (33.4484, -112.0740), "tucson, az": (32.2226, -110.9747),
    "scottsdale, az": (33.4942, -111.9261), "tempe, az": (33.4255, -111.9400),
    "flagstaff, az": (35.1983, -111.6513), "sedona, az": (34.8697, -111.7610),
    # Arkansas
    "little rock, ar": (34.7465, -92.2896), "fayetteville, ar": (36.0626, -94.1574),
    # California
    "los angeles, ca": (34.0522, -118.2437), "san francisco, ca": (37.7749, -122.4194),
    "san diego, ca": (32.7157, -117.1611), "san jose, ca": (37.3382, -121.8863),
    "oakland, ca": (37.8044, -122.2712), "sacramento, ca": (38.5816, -121.4944),
    "santa barbara, ca": (34.4208, -119.6982), "berkeley, ca": (37.8716, -122.2727),
    "mccloud, ca": (41.2554, -122.1392), "ukiah, ca": (39.1502, -123.2078),
    "diamond springs, ca": (38.6941, -120.8324), "ashland, or": (42.1946, -122.7095),
    "redding, ca": (40.5865, -122.3917), "santa cruz, ca": (36.9741, -122.0308),
    "monterey, ca": (36.6002, -121.8947), "carmel, ca": (36.5552, -121.9233),
    "fresno, ca": (36.7378, -119.7871), "pasadena, ca": (34.1478, -118.1445),
    "long beach, ca": (33.7701, -118.1937), "santa rosa, ca": (38.4404, -122.7141),
    "napa, ca": (38.2975, -122.2869), "sonoma, ca": (38.2919, -122.4580),
    # Colorado
    "denver, co": (39.7392, -104.9903), "boulder, co": (40.0150, -105.2705),
    "colorado springs, co": (38.8339, -104.8214), "fort collins, co": (40.5853, -105.0844),
    # Connecticut
    "hartford, ct": (41.7658, -72.6734), "new haven, ct": (41.3083, -72.9279),
    # Delaware
    "wilmington, de": (39.7391, -75.5398), "dover, de": (39.1582, -75.5244),
    # Florida
    "miami, fl": (25.7617, -80.1918), "orlando, fl": (28.5383, -81.3792),
    "tampa, fl": (27.9506, -82.4572), "jacksonville, fl": (30.3322, -81.6557),
    "st. augustine, fl": (29.8946, -81.3145), "tallahassee, fl": (30.4383, -84.2807),
    "dunedin, fl": (28.0197, -82.7718), "sarasota, fl": (27.3364, -82.5307),
    "gainesville, fl": (29.6516, -82.3248), "pensacola, fl": (30.4213, -87.2169),
    "st. petersburg, fl": (27.7676, -82.6403), "fort lauderdale, fl": (26.1224, -80.1373),
    "west palm beach, fl": (26.7153, -80.0534), "key west, fl": (24.5551, -81.7800),
    "clearwater, fl": (27.9659, -82.8001), "naples, fl": (26.1420, -81.7948),
    # Georgia
    "atlanta, ga": (33.7490, -84.3880), "savannah, ga": (32.0809, -81.0912),
    "athens, ga": (33.9519, -83.3576), "augusta, ga": (33.4735, -82.0105),
    # Idaho
    "boise, id": (43.6150, -116.2023),
    # Illinois
    "chicago, il": (41.8781, -87.6298), "springfield, il": (39.7817, -89.6501),
    "champaign, il": (40.1164, -88.2434), "peoria, il": (40.6936, -89.5890),
    # Indiana
    "indianapolis, in": (39.7684, -86.1581), "bloomington, in": (39.1653, -86.5264),
    "fort wayne, in": (41.0793, -85.1394), "south bend, in": (41.6764, -86.2520),
    # Iowa
    "des moines, ia": (41.5868, -93.6250), "iowa city, ia": (41.6611, -91.5302),
    "mount vernon, ia": (41.9219, -91.4168), "cedar rapids, ia": (41.9779, -91.6656),
    "dubuque, ia": (42.5006, -90.6648), "davenport, ia": (41.5236, -90.5776),
    # Kansas
    "kansas city, ks": (39.1141, -94.6275), "wichita, ks": (37.6872, -97.3301),
    "hesston, ks": (38.1383, -97.4314), "lawrence, ks": (38.9717, -95.2353),
    "topeka, ks": (39.0473, -95.6752),
    # Kentucky
    "louisville, ky": (38.2527, -85.7585), "lexington, ky": (38.0406, -84.5037),
    # Louisiana
    "new orleans, la": (29.9511, -90.0715), "baton rouge, la": (30.4515, -91.1871),
    # Maine
    "portland, me": (43.6591, -70.2568), "bangor, me": (44.8016, -68.7712),
    # Maryland
    "baltimore, md": (39.2904, -76.6122), "annapolis, md": (38.9784, -76.4922),
    # Massachusetts
    "boston, ma": (42.3601, -71.0589), "cambridge, ma": (42.3736, -71.1097),
    "worcester, ma": (42.2626, -71.8023), "northampton, ma": (42.3251, -72.6412),
    # Michigan
    "detroit, mi": (42.3314, -83.0458), "ann arbor, mi": (42.2808, -83.7430),
    "grand rapids, mi": (42.9634, -85.6681), "lansing, mi": (42.7325, -84.5555),
    "muskegon, mi": (43.2342, -86.2484), "grayling, mi": (44.6614, -84.7147),
    "saline, mi": (42.1670, -83.7816), "traverse city, mi": (44.7631, -85.6206),
    "kalamazoo, mi": (42.2917, -85.5872), "saginaw, mi": (43.4195, -83.9508),
    "marquette, mi": (46.5436, -87.3954), "petoskey, mi": (45.3733, -84.9553),
    # Minnesota
    "minneapolis, mn": (44.9778, -93.2650), "st. paul, mn": (44.9537, -93.0900),
    "duluth, mn": (46.7867, -92.1005),
    # Mississippi
    "jackson, ms": (32.2988, -90.1848), "oxford, ms": (34.3665, -89.5192),
    # Missouri
    "st. louis, mo": (38.6270, -90.1994), "kansas city, mo": (39.0997, -94.5786),
    "columbia, mo": (38.9517, -92.3341), "springfield, mo": (37.2090, -93.2923),
    "jefferson city, mo": (38.5767, -92.1735), "branson, mo": (36.6436, -93.2185),
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
    # New Mexico
    "albuquerque, nm": (35.0844, -106.6504), "santa fe, nm": (35.6870, -105.9378),
    # New York
    "new york, ny": (40.7128, -74.0060), "brooklyn, ny": (40.6782, -73.9442),
    "buffalo, ny": (42.8864, -78.8784), "albany, ny": (42.6526, -73.7562),
    "rochester, ny": (43.1566, -77.6088), "syracuse, ny": (43.0481, -76.1474),
    "ithaca, ny": (42.4440, -76.5019), "woodstock, ny": (42.0409, -74.1182),
    # North Carolina
    "charlotte, nc": (35.2271, -80.8431), "raleigh, nc": (35.7796, -78.6382),
    "durham, nc": (35.9940, -78.8986), "asheville, nc": (35.5951, -82.5515),
    "wilmington, nc": (34.2257, -77.9447), "greensboro, nc": (36.0726, -79.7920),
    # North Dakota
    "fargo, nd": (46.8772, -96.7898),
    # Ohio
    "columbus, oh": (39.9612, -82.9988), "cleveland, oh": (41.4993, -81.6944),
    "cincinnati, oh": (39.1031, -84.5120), "dayton, oh": (39.7589, -84.1916),
    "akron, oh": (41.0814, -81.5190), "toledo, oh": (41.6528, -83.5379),
    # Oklahoma
    "oklahoma city, ok": (35.4676, -97.5164), "tulsa, ok": (36.1540, -95.9928),
    # Oregon
    "portland, or": (45.5152, -122.6784), "eugene, or": (44.0521, -123.0868),
    "ashland, or": (42.1946, -122.7095), "bend, or": (44.0582, -121.3153),
    "salem, or": (44.9429, -123.0351), "medford, or": (42.3265, -122.8756),
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
    # Utah
    "salt lake city, ut": (40.7608, -111.8910), "provo, ut": (40.2338, -111.6585),
    # Vermont
    "burlington, vt": (44.4759, -73.2121), "montpelier, vt": (44.2601, -72.5754),
    # Virginia
    "richmond, va": (37.5407, -77.4360), "norfolk, va": (36.8508, -76.2859),
    "charlottesville, va": (38.0293, -78.4767), "virginia beach, va": (36.8529, -75.9780),
    "great falls, va": (39.0054, -77.2883), "roanoke, va": (37.2710, -79.9414),
    "alexandria, va": (38.8048, -77.0469), "arlington, va": (38.8816, -77.0910),
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


def geocode_location(location_str, verbose=False):
    """Convert location string to coordinates."""
    if not location_str:
        return None

    loc_lower = location_str.lower().strip()

    # Check lookup table (exact match)
    if loc_lower in US_CITY_COORDS:
        return US_CITY_COORDS[loc_lower]

    # Try partial matching - check if city name matches
    city_name = loc_lower.split(',')[0].strip()
    for key, coords in US_CITY_COORDS.items():
        key_city = key.split(',')[0].strip()
        if city_name == key_city:
            return coords

    # Fall back to geocoding API with cache
    if location_str in _geocode_cache:
        return _geocode_cache[location_str]

    try:
        geolocator = Nominatim(user_agent="newsletter-builder-hoh")
        location = geolocator.geocode(location_str + ", USA", timeout=5)  # Increased timeout
        if location:
            coords = (location.latitude, location.longitude)
            _geocode_cache[location_str] = coords
            if verbose:
                print(f"  [API] {location_str}")
            return coords
    except Exception as e:
        if verbose:
            print(f"  [ERROR] {location_str}: {e}")

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
    failed = []
    for show in shows:
        location = show.get('location', '')
        if location:
            result = geocode_location(location, verbose=True)
            if result:
                coords.append(result)
                print(f"  ✓ {location}")
            else:
                failed.append(location)
                print(f"  ✗ {location} - NOT FOUND")

    if not coords:
        print("Could not geocode any locations!")
        input("\nPress Enter to close...")
        return

    print(f"\nMapped {len(coords)} of {len(shows)} locations")
    if failed:
        print(f"\n⚠️  {len(failed)} locations could not be geocoded:")
        for loc in failed:
            print(f"   - {loc}")
        print("\nTip: Add missing cities to US_CITY_COORDS in update_tour_map.py")

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

        # Check if there are changes to commit
        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=PROJECT_ROOT, capture_output=True, text=True
        )

        if not status.stdout.strip():
            print("Map unchanged - nothing to push.")
        else:
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
        print(f"Git error: {error_msg}")

    print("\n" + "=" * 50)
    print("  DONE!")
    print("=" * 50)
    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()

"""
Scraper for House of Hamill tour dates via Bandsintown API.
"""

import requests
from datetime import datetime
from config import BANDSINTOWN_API_KEY, ARTIST_NAME, MAX_SHOWS_TO_DISPLAY


def get_upcoming_shows():
    """
    Fetch upcoming shows from Bandsintown API.
    Returns a list of show dicts with venue, date, location, and ticket URL.
    """
    if not BANDSINTOWN_API_KEY:
        print("Warning: No Bandsintown API key configured.")
        print("Add your key to config.py to enable show fetching.")
        return []

    # URL-encode the artist name
    artist_encoded = requests.utils.quote(ARTIST_NAME)
    
    url = f"https://rest.bandsintown.com/artists/{artist_encoded}/events"
    params = {
        "app_id": BANDSINTOWN_API_KEY,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        events = response.json()
    except requests.RequestException as e:
        print(f"Error fetching shows: {e}")
        return []
    except ValueError:
        print("Error parsing Bandsintown response")
        return []

    shows = []
    for event in events[:MAX_SHOWS_TO_DISPLAY]:
        # Parse the datetime
        dt_str = event.get("datetime", "")
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            date_formatted = dt.strftime("%b %d")  # e.g., "Mar 15"
            day_of_week = dt.strftime("%a")        # e.g., "Sat"
        except (ValueError, AttributeError):
            date_formatted = "TBA"
            day_of_week = ""

        venue = event.get("venue", {})
        
        # Build location string
        city = venue.get("city", "")
        region = venue.get("region", "")
        country = venue.get("country", "")
        
        if country == "United States" or country == "US":
            location = f"{city}, {region}" if region else city
        else:
            location = f"{city}, {country}" if country else city

        # Get ticket URL - prefer the offers array, fall back to event URL
        ticket_url = ""
        offers = event.get("offers", [])
        if offers:
            ticket_url = offers[0].get("url", "")
        if not ticket_url:
            ticket_url = event.get("url", "")

        show = {
            "date": date_formatted,
            "day": day_of_week,
            "venue": venue.get("name", "TBA"),
            "location": location,
            "ticket_url": ticket_url,
            "datetime": dt if dt_str else None,
        }
        shows.append(show)

    return shows


def format_shows_for_display(shows):
    """Format shows list for terminal display."""
    if not shows:
        return "No upcoming shows found."
    
    lines = []
    for show in shows:
        lines.append(f"{show['day']} {show['date']} | {show['venue']} | {show['location']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Fetching upcoming shows for {ARTIST_NAME}...\n")
    shows = get_upcoming_shows()
    if shows:
        print(format_shows_for_display(shows))
    else:
        print("No shows found (or API key not configured)")

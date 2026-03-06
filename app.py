import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
import sys

try:
    import folium
    from streamlit_folium import st_folium
    from geopy.geocoders import Nominatim
    import requests as _requests
    MAP_LIBS = True
except ImportError:
    MAP_LIBS = False

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from predict import get_ranked_places, load_data
from itinerary import build_itinerary

# =========================
# COLUMN AUTO-DETECTION
# =========================
def detect_columns(columns):
    """Fuzzy-match uploaded column names to required fields.
    Each source column is assigned to at most one field (first best match wins).
    """
    # Order matters: more specific fields first to avoid collisions
    patterns = {
        'review_count': ['number of google review', 'review count', 'num review', 'number of review', 'reviews in'],
        'rating':       ['google review rating', 'review rating', 'rating', 'score', 'stars'],
        'visit_time':   ['time needed', 'visit time', 'time in hrs', 'duration', 'hours needed'],
        'city':         ['city', 'town', 'district'],
        'state':        ['state', 'province', 'region'],
        'place_name':   ['place name', 'name', 'attraction', 'destination', 'site'],
        'type':         ['type', 'category', 'kind'],
        'significance': ['significance', 'important'],
        'fee':          ['entrance fee', 'fee', 'price', 'cost', 'ticket'],
        'maps_url':     ['maps', 'map url', 'google maps', 'location url', 'place url'],
    }
    detected = {}
    used_columns = set()
    col_lower = {c.lower(): c for c in columns}
    for field, keywords in patterns.items():
        for col_l, col_orig in col_lower.items():
            if col_orig in used_columns:
                continue
            for kw in keywords:
                if kw in col_l:
                    detected[field] = col_orig
                    used_columns.add(col_orig)
                    break
            if field in detected:
                break
    return detected

# =========================
# GEOCODING & MAP HELPERS
# =========================
@st.cache_data(show_spinner=False)
def geocode_place(place_name: str, city: str):
    """Return (lat, lon) for a place via Nominatim. Returns (None, None) on failure."""
    if not MAP_LIBS:
        return None, None
    try:
        geolocator = Nominatim(user_agent="travel_maker_app_v1", timeout=8)
        time.sleep(1)  # Nominatim rate-limit: 1 req/sec
        location = geolocator.geocode(f"{place_name}, {city}, India")
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass
    return None, None


@st.cache_data(show_spinner=False)
def get_road_route(lat1: float, lon1: float, lat2: float, lon2: float):
    """
    Fetch the actual road route between two points using OSRM (free, no API key).
    Returns (road_polyline_coords [[lat,lon],...], distance_km, duration_min)
    or (None, None, None) on failure.
    """
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}"
        f"?overview=full&geometries=geojson&steps=false"
    )
    try:
        resp = _requests.get(url, timeout=12)
        data = resp.json()
        if data.get("code") == "Ok":
            route = data["routes"][0]
            dist_km = round(route["distance"] / 1000.0, 1)
            dur_min = round(route["duration"] / 60.0, 0)
            # GeoJSON coords are [lon, lat] — flip to [lat, lon] for folium
            coords = [[c[1], c[0]] for c in route["geometry"]["coordinates"]]
            return coords, dist_km, int(dur_min)
    except Exception:
        pass
    return None, None, None


_DAY_COLORS = ["red", "blue", "green", "purple", "orange",
               "darkred", "darkblue", "darkgreen", "cadetblue", "lightred"]


_LINE_COLORS = {
    "red": "#e74c3c", "blue": "#2980b9", "green": "#27ae60",
    "purple": "#8e44ad", "orange": "#e67e22", "darkred": "#922b21",
    "darkblue": "#1a5276", "darkgreen": "#1e8449",
    "cadetblue": "#2e86c1", "lightred": "#f1948a",
}
_CIRCLE_COLORS = {
    "red": "#e74c3c", "blue": "#2980b9", "green": "#27ae60",
    "purple": "#8e44ad", "orange": "#e67e22", "darkred": "#922b21",
    "darkblue": "#1a5276", "darkgreen": "#1e8449",
    "cadetblue": "#2e86c1", "lightred": "#f1948a",
}


def _draw_route(m, coords, valid_places, color, label_prefix=""):
    """
    Draw numbered markers + road-following polylines with distance labels.
    Uses OSRM for actual road routes; falls back to straight line if OSRM fails.
    Returns a list of distance-row dicts.
    """
    hex_color = _LINE_COLORS.get(color, "#e74c3c")
    circle_bg = _CIRCLE_COLORS.get(color, "#e74c3c")
    distances = []

    # --- Markers ---
    for idx, (place, (lat, lon)) in enumerate(zip(valid_places, coords), 1):
        maps_url = place.get("maps_url", "")
        link_html = (
            f'<a href="{maps_url}" target="_blank">📍 Open in Google Maps</a>'
            if maps_url else ""
        )
        score_line = f"🎯 ML Score: {place['ml_score']:.3f}<br>" if "ml_score" in place else ""
        popup_html = f"""
        <div style="font-family:sans-serif;min-width:170px">
            <b>{label_prefix}Stop {idx}: {place['place_name']}</b><br>
            ⭐ {place['rating']:.1f}/5 &nbsp;|&nbsp; ⏱️ {place['visit_time']:.1f} hrs<br>
            {score_line}{link_html}
        </div>"""
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{label_prefix}Stop {idx}: {place['place_name']}",
            icon=folium.DivIcon(
                html=(
                    f'<div style="background:{circle_bg};color:white;'
                    f'width:28px;height:28px;border-radius:50%;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-weight:bold;font-size:13px;border:2px solid white;'
                    f'box-shadow:0 2px 4px rgba(0,0,0,.45)">{idx}</div>'
                ),
                icon_size=(28, 28),
                icon_anchor=(14, 14),
            ),
        ).add_to(m)

    # --- Road routes between consecutive stops ---
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]

        road_coords, dist_km, dur_min = get_road_route(p1[0], p1[1], p2[0], p2[1])

        if road_coords:
            folium.PolyLine(
                road_coords,
                color=hex_color,
                weight=4,
                opacity=0.85,
                tooltip=f"🚗 {dist_km:.1f} km · ~{dur_min} min by road",
            ).add_to(m)
            mid_idx = len(road_coords) // 2
            label_pos = road_coords[mid_idx]
            label_text = f"🚗 {dist_km} km · {dur_min} min"
        else:
            # Fallback: straight dashed line with geodesic distance
            from geopy.distance import geodesic as _geodesic
            dist_km = round(_geodesic(p1, p2).km, 1)
            dur_min = None
            folium.PolyLine(
                [p1, p2],
                color=hex_color,
                weight=3,
                opacity=0.55,
                dash_array="8 6",
                tooltip=f"📏 {dist_km:.1f} km (straight line, road data unavailable)",
            ).add_to(m)
            label_pos = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
            label_text = f"📏 {dist_km} km"

        folium.Marker(
            label_pos,
            icon=folium.DivIcon(
                html=(
                    f'<div style="background:white;border:1px solid {hex_color};'
                    f'border-radius:10px;padding:2px 8px;font-size:11px;'
                    f'font-weight:600;color:{hex_color};white-space:nowrap;'
                    f'box-shadow:0 1px 3px rgba(0,0,0,.2)">{label_text}</div>'
                ),
                icon_size=(140, 22),
                icon_anchor=(70, 11),
            ),
        ).add_to(m)

        row = {
            "From": valid_places[i]["place_name"],
            "To": valid_places[i + 1]["place_name"],
            "Distance (km)": dist_km,
        }
        if dur_min is not None:
            row["Drive Time (min)"] = dur_min
        distances.append(row)

    return distances


def build_places_map(places: list, city: str, day_groups: dict = None):
    """
    Build a folium map with numbered markers connected by distance-labelled route lines.

    - day_groups=None  → all places in one ranked route (Explore mode)
    - day_groups=dict  → each day drawn in a different colour (Itinerary mode)

    Returns (folium.Map, list-of-distance-dicts).
    """
    city_lat, city_lon = geocode_place(city, city)
    center = [city_lat, city_lon] if city_lat else [20.5937, 78.9629]
    zoom = 12 if city_lat else 5
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

    all_distances = []

    groups = day_groups if day_groups else {"": places}

    for day_idx, (day_name, group_places) in enumerate(groups.items()):
        color = _DAY_COLORS[day_idx % len(_DAY_COLORS)]
        coords, valid_places = [], []
        for place in group_places:
            lat, lon = geocode_place(place["place_name"], city)
            if lat:
                coords.append((lat, lon))
                valid_places.append(place)

        rows = _draw_route(m, coords, valid_places, color, label_prefix=f"{day_name} " if day_name else "")
        for r in rows:
            if day_name:
                r["Day"] = day_name
        all_distances.extend(rows)

    return m, all_distances


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Travel Maker - ML Itinerary Generator",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .place-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #FF6B6B;
    }
    .place-name {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2C3E50;
    }
    .place-details {
        color: #7F8C8D;
        font-size: 0.95rem;
    }
    .ml-score {
        background-color: #FF6B6B;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .day-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="main-header">🧳 Travel Maker</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ML-Powered Itinerary Generator for Indian Cities</div>', unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

# Load data to get available cities
@st.cache_data
def get_available_cities():
    df = load_data()
    return sorted(df['city'].unique().tolist())

cities = get_available_cities()

# Mode selection
mode = st.sidebar.radio(
    "Choose Mode:",
    ["🔍 Explore Places", "📅 Generate Itinerary", "📤 Upload & Train"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Configuration")

model_type = st.sidebar.selectbox(
    "Model:",
    ["XGBoost Regressor"],
    index=0
)

n_estimators = st.sidebar.slider("Trees (n_estimators)", 50, 500, 200, step=50)
max_depth = st.sidebar.slider("Max Depth", 2, 10, 6)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.30, 0.05, step=0.01, format="%.2f")

st.sidebar.markdown("---")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def get_full_data():
    return load_data()

df_full = get_full_data()

# =========================
# MODE 1: EXPLORE PLACES
# =========================
if mode == "🔍 Explore Places":
    st.header("🔍 Explore Top-Rated Places")

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_city = st.selectbox(
            "Select a city:",
            cities,
            index=0 if cities else None
        )

    with col2:
        top_k = st.slider("Number of places to show:", 5, 30, 10)

    if st.button("🔎 Search Places", type="primary", use_container_width=True):
        with st.spinner(f"Finding top places in {selected_city}..."):
            places = get_ranked_places(selected_city, top_k=top_k)

            if not places:
                st.error(f"No places found for {selected_city}. Please try another city.")
            else:
                st.success(f"Found {len(places)} amazing places in {selected_city}!")

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)

                avg_rating = sum(p['rating'] for p in places) / len(places)
                total_time = sum(p['visit_time'] for p in places)
                max_score = max(p['ml_score'] for p in places)

                with col1:
                    st.metric("Average Rating", f"⭐ {avg_rating:.2f}")
                with col2:
                    st.metric("Total Visit Time", f"⏱️ {total_time:.1f} hrs")
                with col3:
                    st.metric("Top ML Score", f"🎯 {max_score:.3f}")
                with col4:
                    st.metric("Places Found", f"📍 {len(places)}")

                st.markdown("---")

                # Display places
                for idx, place in enumerate(places, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"""
                            <div class="place-card">
                                <div class="place-name">
                                    {idx}. {place['place_name']}
                                </div>
                                <div class="place-details">
                                    ⭐ Rating: {place['rating']:.1f}/5 |
                                    ⏱️ Visit Time: {place['visit_time']:.1f} hours |
                                    <span class="ml-score">ML Score: {place['ml_score']:.3f}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            # Get additional details from full dataset
                            place_details = df_full[df_full['place_name'] == place['place_name']]
                            if not place_details.empty:
                                place_info = place_details.iloc[0]
                                fee = place_info.get('fee', 'N/A')
                                type_cat = place_info.get('type', 'N/A')

                                with st.expander("📋 Details"):
                                    st.write(f"**Type:** {type_cat}")
                                    st.write(f"**Fee:** ₹{fee}")
                                    st.write(f"**State:** {place_info.get('state', 'N/A')}")
                                    st.write(f"**Significance:** {place_info.get('significance', 'N/A')}")

                            maps_url = place.get("maps_url", "")
                            if maps_url:
                                st.link_button("📍 Open in Maps", maps_url, use_container_width=True)

                # Map view
                if MAP_LIBS:
                    st.markdown("---")
                    with st.expander("🗺️ View All on Map + Distances", expanded=False):
                        with st.spinner("Geocoding places… (first load may take a moment)"):
                            m, distances = build_places_map(places, selected_city)
                        st_folium(m, width="100%", height=520, returned_objects=[])
                        if distances:
                            st.markdown("**📏 Route distances (in ranked order):**")
                            base_cols = ["From", "To", "Distance (km)"]
                            if any("Drive Time (min)" in d for d in distances):
                                base_cols.append("Drive Time (min)")
                            dist_df = pd.DataFrame(distances).reindex(columns=base_cols)
                            st.dataframe(dist_df, use_container_width=True, hide_index=True)
                            total_km = sum(d["Distance (km)"] for d in distances)
                            total_min = sum(d.get("Drive Time (min)", 0) for d in distances)
                            c1, c2 = st.columns(2)
                            c1.metric("Total Road Distance", f"{total_km:.1f} km")
                            if total_min:
                                c2.metric("Total Drive Time", f"{int(total_min)} min")

# =========================
# MODE 2: GENERATE ITINERARY
# =========================
elif mode == "📅 Generate Itinerary":
    st.header("📅 Generate Custom Itinerary")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        selected_city = st.selectbox(
            "Select destination city:",
            cities,
            index=0 if cities else None
        )

    with col2:
        days = st.number_input("Number of days:", min_value=1, max_value=7, value=2)

    with col3:
        hours_per_day = st.number_input(
            "Hours per day:",
            min_value=4.0,
            max_value=12.0,
            value=8.0,
            step=0.5
        )

    if st.button("🗓️ Generate Itinerary", type="primary", use_container_width=True):
        with st.spinner(f"Creating your {days}-day itinerary for {selected_city}..."):
            itinerary = build_itinerary(selected_city, days, hours_per_day)

            if "error" in itinerary:
                st.error(itinerary["error"])
            else:
                st.success(f"✨ Your {days}-day itinerary is ready!")

                # Calculate stats
                total_places = sum(len(places) for places in itinerary.values())
                total_time = 0

                for day_places in itinerary.values():
                    for place in day_places:
                        total_time += place.get('visit_time', 0)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Places", f"📍 {total_places}")
                with col2:
                    st.metric("Total Time", f"⏱️ {total_time:.1f} hrs")
                with col3:
                    st.metric("Days", f"📅 {days}")

                st.markdown("---")

                # Display day-wise itinerary
                for day_name, places in itinerary.items():
                    st.markdown(f'<div class="day-header">{day_name} - {len(places)} Places</div>',
                              unsafe_allow_html=True)

                    if not places:
                        st.info("No places scheduled for this day.")
                        continue

                    day_total_time = sum(p.get('visit_time', 0) for p in places)
                    st.write(f"**Total time for this day:** {day_total_time:.1f} hours")

                    for idx, place in enumerate(places, 1):
                        col1, col2, col3 = st.columns([3, 1, 1])

                        with col1:
                            st.markdown(f"**{idx}. {place['place_name']}**")

                        with col2:
                            st.write(f"⭐ {place['rating']:.1f}/5")

                        with col3:
                            st.write(f"⏱️ {place['visit_time']:.1f} hrs")

                        # Get additional details
                        place_details = df_full[df_full['place_name'] == place['place_name']]
                        if not place_details.empty:
                            place_info = place_details.iloc[0]
                            with st.expander("More info"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write(f"**Type:** {place_info.get('type', 'N/A')}")
                                    st.write(f"**Fee:** ₹{place_info.get('fee', 'N/A')}")
                                with col_b:
                                    st.write(f"**ML Score:** {place.get('ml_score', 0):.3f}")
                                    st.write(f"**Significance:** {place_info.get('significance', 'N/A')}")

                        maps_url = place.get("maps_url", "")
                        if maps_url:
                            st.link_button("📍 Open in Maps", maps_url)

                        st.markdown("---")

                # Full trip map
                if MAP_LIBS:
                    st.markdown("---")
                    with st.expander("🗺️ View Full Trip on Map + Distances", expanded=False):
                        with st.spinner("Geocoding places… (first load may take a moment)"):
                            m, distances = build_places_map([], selected_city, day_groups=itinerary)
                        st_folium(m, width="100%", height=540, returned_objects=[])
                        if distances:
                            st.markdown("**📏 Distances between consecutive stops:**")
                            base_cols = ["From", "To", "Distance (km)"]
                            if any("Day" in d for d in distances):
                                base_cols = ["Day"] + base_cols
                            if any("Drive Time (min)" in d for d in distances):
                                base_cols.append("Drive Time (min)")
                            dist_df = pd.DataFrame(distances).reindex(columns=base_cols)
                            st.dataframe(dist_df, use_container_width=True, hide_index=True)
                            total_km = sum(d["Distance (km)"] for d in distances)
                            total_min = sum(d.get("Drive Time (min)", 0) for d in distances)
                            c1, c2 = st.columns(2)
                            c1.metric("Total Road Distance", f"{total_km:.1f} km")
                            if total_min:
                                c2.metric("Total Drive Time", f"{int(total_min)} min")

# =========================
# MODE 3: UPLOAD & TRAIN
# =========================
elif mode == "📤 Upload & Train":
    st.header("📤 Upload Dataset & Train Model")

    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)

        st.success(f"Loaded **{len(df_upload):,} rows** and **{len(df_upload.columns)} columns**")

        with st.expander("Preview uploaded data", expanded=True):
            st.dataframe(df_upload.head(10), use_container_width=True)

        st.markdown("---")
        st.subheader("Column Mapping")
        st.caption("Auto-detected mappings are pre-filled. Adjust if any are wrong.")

        detected = detect_columns(df_upload.columns.tolist())
        col_options = ["(skip)"] + df_upload.columns.tolist()

        REQUIRED_FIELDS = {
            "city":         "City name",
            "place_name":   "Place / attraction name",
            "type":         "Place type / category",
            "significance": "Significance / importance level",
            "rating":       "Rating (numeric, e.g. 4.2)",
            "review_count": "Number of reviews",
            "visit_time":   "Visit duration in hours",
        }
        OPTIONAL_FIELDS = {
            "state": "State / province (optional)",
            "fee":   "Entrance fee (optional)",
        }

        mapping = {}
        left, right = st.columns(2)
        all_fields = list(REQUIRED_FIELDS.items()) + list(OPTIONAL_FIELDS.items())

        for i, (field, desc) in enumerate(all_fields):
            col = left if i % 2 == 0 else right
            default = detected.get(field, "(skip)")
            idx = col_options.index(default) if default in col_options else 0
            label = f"**{field}** — {desc}" if field in REQUIRED_FIELDS else f"{field} — {desc}"
            with col:
                mapping[field] = st.selectbox(label, col_options, index=idx, key=f"map_{field}")

        unmapped = [f for f in REQUIRED_FIELDS if mapping.get(f) == "(skip)"]
        if unmapped:
            st.warning(f"Please map required fields: {', '.join(unmapped)}")
        else:
            st.markdown("---")
            st.subheader("Train Model")
            st.caption("Uses the hyperparameters set in the Model Configuration section of the sidebar.")

            if st.button("🚀 Train on Uploaded Data", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    from sklearn.preprocessing import LabelEncoder
                    from xgboost import XGBRegressor

                    status_text.text("Applying column mapping...")
                    progress_bar.progress(10)

                    # Build df_t by extracting each mapped column individually.
                    # This avoids the rename-collision bug where two fields pointing
                    # to the same source column cause one field to disappear.
                    df_t = pd.DataFrame(index=df_upload.index)
                    for field, col in mapping.items():
                        if col != "(skip)":
                            df_t[field] = df_upload[col].values

                    # Fill missing values
                    status_text.text("Cleaning data...")
                    progress_bar.progress(25)

                    df_t["rating"] = pd.to_numeric(df_t["rating"], errors="coerce")
                    df_t["review_count"] = pd.to_numeric(df_t["review_count"], errors="coerce").fillna(0)
                    df_t["visit_time"] = pd.to_numeric(df_t["visit_time"], errors="coerce")
                    df_t["rating"] = df_t["rating"].fillna(df_t["rating"].mean())
                    df_t["visit_time"] = df_t["visit_time"].fillna(df_t["visit_time"].median())
                    df_t["significance"] = df_t["significance"].fillna("Local") if "significance" in df_t.columns else "Local"

                    df_t["log_reviews"] = np.log1p(df_t["review_count"])

                    status_text.text("Encoding categorical features...")
                    progress_bar.progress(45)

                    le_city = LabelEncoder()
                    df_t["city_encoded"] = le_city.fit_transform(df_t["city"].astype(str))
                    le_type = LabelEncoder()
                    df_t["type_encoded"] = le_type.fit_transform(df_t["type"].astype(str))
                    le_sig = LabelEncoder()
                    df_t["sig_encoded"] = le_sig.fit_transform(df_t["significance"].astype(str))

                    df_t["target_score"] = (
                        df_t["rating"] * 0.6 +
                        df_t["log_reviews"] * 0.3 +
                        (1 / (df_t["visit_time"] + 1)) * 0.1
                    )

                    features = ["city_encoded", "type_encoded", "sig_encoded", "rating", "log_reviews", "visit_time"]
                    X = df_t[features]
                    y = df_t["target_score"]

                    status_text.text(f"Training XGBoost ({n_estimators} trees, depth {max_depth}, lr {learning_rate:.2f})...")
                    progress_bar.progress(60)

                    model = XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42
                    )
                    model.fit(X, y)

                    status_text.text("Saving model and dataset...")
                    progress_bar.progress(85)

                    MODEL_PATH = Path("models/xgb_ranker.pkl")
                    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump({
                        "model": model,
                        "label_encoders": {"city": le_city, "type": le_type, "significance": le_sig},
                        "features": features
                    }, MODEL_PATH)

                    # Save cleaned dataset so predict.py can use it
                    DATASET_OUT = Path("dataset/indian_place.xlsx")
                    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
                    df_t.to_excel(DATASET_OUT, index=False)

                    progress_bar.progress(100)
                    status_text.text("")
                    st.success(
                        f"Model trained on **{len(df_t):,} rows** from **{uploaded_file.name}**  \n"
                        f"Trees: {n_estimators} | Depth: {max_depth} | LR: {learning_rate:.2f}  \n"
                        f"Dataset saved — switch to Explore or Itinerary mode to use it."
                    )
                    st.cache_data.clear()

                except Exception as e:
                    progress_bar.progress(0)
                    status_text.text("")
                    st.error(f"Training failed: {e}")

# =========================
# FOOTER
# =========================
if st.sidebar.button("🚀 Train Model", use_container_width=True):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    try:
        status_text.text("Loading dataset...")
        progress_bar.progress(10)

        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBRegressor

        DATA_PATH = Path("dataset/indian_place.xlsx")
        MODEL_PATH = Path("models/xgb_ranker.pkl")

        df = pd.read_excel(DATA_PATH)
        df = df.rename(columns={
            "City": "city", "State": "state", "Name": "place_name",
            "Type": "type", "Significance": "significance",
            "Google review rating": "rating",
            "Number of google review in lakhs": "review_count",
            "time needed to visit in hrs": "visit_time",
            "Entrance Fee in INR": "fee"
        })
        progress_bar.progress(25)
        status_text.text("Preprocessing data...")

        df["rating"] = df["rating"].fillna(df["rating"].mean())
        df["review_count"] = df["review_count"].fillna(0)
        df["visit_time"] = df["visit_time"].fillna(df["visit_time"].median())
        df["significance"] = df["significance"].fillna("Local")
        df["log_reviews"] = np.log1p(df["review_count"])

        le_city = LabelEncoder()
        df["city_encoded"] = le_city.fit_transform(df["city"].astype(str))
        le_type = LabelEncoder()
        df["type_encoded"] = le_type.fit_transform(df["type"].astype(str))
        le_sig = LabelEncoder()
        df["sig_encoded"] = le_sig.fit_transform(df["significance"].astype(str))

        df["target_score"] = (
            df["rating"] * 0.6 +
            df["log_reviews"] * 0.3 +
            (1 / (df["visit_time"] + 1)) * 0.1
        )
        progress_bar.progress(50)
        status_text.text(f"Training {model_type}...")

        features = ["city_encoded", "type_encoded", "sig_encoded", "rating", "log_reviews", "visit_time"]
        X = df[features]
        y = df["target_score"]

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X, y)
        progress_bar.progress(85)
        status_text.text("Saving model...")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": model,
            "label_encoders": {"city": le_city, "type": le_type, "significance": le_sig},
            "features": features
        }, MODEL_PATH)

        progress_bar.progress(100)
        status_text.text("Done!")
        st.sidebar.success(f"Model trained! ({n_estimators} trees, depth {max_depth}, lr {learning_rate:.2f})")
        st.cache_data.clear()

    except Exception as e:
        progress_bar.progress(0)
        status_text.text("")
        st.sidebar.error(f"Training failed: {e}")

st.sidebar.markdown("---")
st.sidebar.info("""
**About Travel Maker**

This app uses XGBoost machine learning to:
- Rank tourist destinations
- Generate optimal itineraries
- Help you explore India efficiently

**Tech Stack:**
- XGBoost for ranking
- Streamlit for UI
- Python for backend

**Model Features:**
- City, Type, Significance
- Google Ratings & Reviews
- Visit Time Estimation
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using ML")

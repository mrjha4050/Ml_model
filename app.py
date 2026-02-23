import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from predict import get_ranked_places, load_data
from itinerary import build_itinerary

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
    ["🔍 Explore Places", "📅 Generate Itinerary"],
    index=0
)

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

                        st.markdown("---")

# =========================
# FOOTER
# =========================
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

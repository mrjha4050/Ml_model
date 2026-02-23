# 🧳 Travel Maker - ML-Powered Itinerary Generator

An intelligent travel recommendation system that uses Machine Learning to rank tourist destinations and generate optimized day-wise itineraries for Indian cities.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Why XGBoost?](#why-xgboost)
- [What I Built](#what-i-built)
- [Dataset](#dataset)
- [Approach & Methodology](#approach--methodology)
- [Model Architecture](#model-architecture)
- [Pros & Cons](#pros--cons)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

Travel planning is time-consuming and overwhelming, especially when visiting a new city with hundreds of attractions. This project solves that problem by:

1. **Ranking tourist places** using ML based on ratings, popularity, and time requirements
2. **Generating optimized itineraries** that fit within your trip duration
3. **Providing a user-friendly interface** to explore recommendations

The system uses **XGBoost (Extreme Gradient Boosting)**, a powerful ensemble learning algorithm, to predict the "quality score" of each tourist destination.

---

## 🤔 Why XGBoost?

I chose **XGBoost** for this project based on the following considerations:

### Reasons for Choosing XGBoost:

1. **Handles Mixed Data Types**: Our dataset contains both numerical (ratings, visit time) and categorical features (city, type, significance)
2. **Robust to Missing Data**: XGBoost has built-in handling for missing values, which is common in tourism data
3. **Feature Importance**: Provides clear insights into what makes a place worth visiting
4. **Fast Training & Inference**: Essential for real-time recommendations
5. **Regularization**: L1 and L2 regularization prevents overfitting on small datasets
6. **Non-linear Relationships**: Captures complex patterns (e.g., very high ratings + many reviews = must-visit)
7. **Industry Standard**: Used by Airbnb, Booking.com, and other travel platforms

### Alternatives Considered:

- **Linear Regression**: Too simple, assumes linear relationships
- **Random Forest**: Good but slower than XGBoost, less accurate
- **Deep Learning**: Overkill for tabular data with ~1000 rows, requires more data
- **Collaborative Filtering**: Would require user interaction data (not available)

---

## 🔨 What I Built

This project consists of three main components:

### 1. **ML Model (XGBoost Ranker)**
   - Trained on Indian tourist destinations dataset
   - Predicts a "quality score" for each place
   - Uses 6 features: city, type, significance, rating, review count, visit time

### 2. **Itinerary Generator**
   - Takes city name and number of days as input
   - Automatically schedules places into day-wise plans
   - Respects time constraints (default: 8 hours per day)

### 3. **Streamlit Web Interface**
   - Interactive UI for exploring recommendations
   - Search places by city
   - Generate custom itineraries
   - View place details (ratings, time needed, entrance fee)

---

## 📊 Dataset

**Source**: Indian tourist destinations dataset (`indian_places.xlsx`)

### Features:
| Column | Description | Example |
|--------|-------------|---------|
| `City` | City name | Jaipur, Delhi, Mumbai |
| `State` | Indian state | Rajasthan, Maharashtra |
| `Name` | Place name | Taj Mahal, Hawa Mahal |
| `Type` | Category | Temple, Fort, Museum, Park |
| `Significance` | Importance level | National, State, Local |
| `Google review rating` | Average rating (0-5) | 4.5 |
| `Number of google review in lakhs` | Review count (in lakhs) | 2.5 (= 250,000 reviews) |
| `time needed to visit in hrs` | Estimated visit duration | 2.5 hours |
| `Entrance Fee in INR` | Entry cost | ₹50, ₹200 |

### Dataset Statistics:
- **Total Places**: Varies by dataset (likely 500-1000+ places)
- **Cities Covered**: Major Indian tourist cities
- **Feature Completeness**: Missing values handled through imputation

---

## 🧠 Approach & Methodology

### 1. **Data Preprocessing**

```python
# Handle missing values
- Ratings: Fill with mean
- Review count: Fill with 0
- Visit time: Fill with median
- Significance: Fill with "Local"

# Feature Engineering
- Log transformation: log(review_count + 1) to handle skewness
- Label encoding: Convert categorical features to numeric
```

### 2. **Target Variable Creation**

Since we don't have explicit "quality labels," I created a **synthetic target score**:

```python
target_score = (rating × 0.6) + (log_reviews × 0.3) + (1/(visit_time+1) × 0.1)
```

**Intuition**:
- **60% weight on rating**: Quality matters most
- **30% weight on popularity**: More reviews = more reliable
- **10% weight on efficiency**: Shorter visits are slightly preferred (more places per day)

### 3. **Model Training**

```python
XGBRegressor(
    n_estimators=200,      # 200 trees for stable predictions
    max_depth=6,           # Moderate depth prevents overfitting
    learning_rate=0.05,    # Slow learning for better generalization
    subsample=0.8,         # 80% data sampling per tree
    colsample_bytree=0.8,  # 80% feature sampling per tree
    random_state=42        # Reproducibility
)
```

### 4. **Prediction & Ranking**

- Load trained model
- Filter places by user's chosen city
- Predict scores for all places
- Sort by ML score (descending)
- Return top-K recommendations

### 5. **Itinerary Generation**

- Greedy algorithm: Pick highest-scoring places first
- Bin-packing: Fit places into days based on visit time
- Constraint: Max 8 hours of sightseeing per day

---

## 🏗️ Model Architecture

```
Input Features (6)
    ↓
[city_encoded, type_encoded, sig_encoded, rating, log_reviews, visit_time]
    ↓
XGBoost Regressor (200 trees, depth=6)
    ↓
ML Score (Predicted Quality)
    ↓
Ranking & Itinerary Generation
```

---

## ⚖️ Pros & Cons

### ✅ Pros

1. **Fast & Efficient**: XGBoost trains in seconds, predicts instantly
2. **Interpretable**: Feature importance shows what matters (rating > reviews > time)
3. **No Cold Start**: Works with zero user history (content-based)
4. **Scalable**: Can easily add more cities/places
5. **Handles Missing Data**: Robust to incomplete information
6. **Works Offline**: Once trained, no API calls needed
7. **Customizable**: Easy to adjust target score weights

### ❌ Cons

1. **No Personalization**: Same recommendations for all users (doesn't learn preferences)
2. **Synthetic Target**: Target score is hand-crafted, not based on real user behavior
3. **Cold Start for New Places**: New places without reviews get low scores
4. **No Diversity**: May recommend similar types of places (e.g., all temples)
5. **Static Data**: Doesn't update with real-time changes (weather, closures)
6. **Simplistic Itinerary**: Doesn't consider travel time between places
7. **No Context Awareness**: Ignores season, user age, interests, budget
8. **Overfitting Risk**: With small datasets, may memorize patterns

### 🎯 When This Approach Works Best:
- First-time visitors to a city
- Users seeking popular, highly-rated attractions
- Short trips (1-3 days)
- Users without strong preferences

### 🚫 When This Approach Falls Short:
- Personalized recommendations (need collaborative filtering)
- Route optimization (need geospatial algorithms)
- Budget-conscious travelers (need constraint optimization)
- Niche interests (need content-based filtering with user profiles)

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd ml_model
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install Streamlit (for UI)
```bash
pip install streamlit
```

---

## 🚀 How to Run

### Option 1: Train the Model

Train the XGBoost model from scratch:

```bash
python src/train.py
```

**Output**: Saves model to `models/xgb_ranker.pkl`

---

### Option 2: Test Predictions (CLI)

Get top places for a city via command line:

```bash
python src/predict.py
```

**Example Output**:
```
Top places in Jaipur:

Hawa Mahal | Rating: 4.5 | Time: 1.5 hrs | Score: 3.421
Amber Fort | Rating: 4.6 | Time: 3.0 hrs | Score: 3.389
City Palace | Rating: 4.4 | Time: 2.5 hrs | Score: 3.198
```

---

### Option 3: Generate Itinerary (CLI)

Create a day-wise itinerary:

```bash
python src/itinerary.py
```

**Example Output**:
```
Itinerary for Jaipur (2 days):

Day 1
  - Hawa Mahal (1.5 hrs)
  - Amber Fort (3.0 hrs)
  - Jantar Mantar (2.0 hrs)

Day 2
  - City Palace (2.5 hrs)
  - Jal Mahal (1.0 hrs)
  - Nahargarh Fort (2.5 hrs)
```

---

### Option 4: Run Streamlit UI (Recommended)

Launch the interactive web interface:

```bash
streamlit run app.py
```

**Features**:
- 🔍 Search places by city
- 📅 Generate multi-day itineraries
- 📊 View ratings, visit times, and ML scores
- 🎨 Beautiful, responsive UI

**Access**: Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
ml_model/
│
├── dataset/
│   └── indian_places.xlsx          # Raw tourism data
│
├── models/
│   └── xgb_ranker.pkl              # Trained XGBoost model
│
├── src/
│   ├── train.py                    # Model training script
│   ├── predict.py                  # Prediction & ranking
│   └── itinerary.py                # Itinerary generation logic
│
├── app.py                          # Streamlit web interface
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation (this file)
└── main.py                         # Sample script
```

---

## 🔮 Future Improvements

### Short-term:
1. **Add more features**: Budget, season, crowd levels
2. **Improve itinerary**: Consider travel time between places using Google Maps API
3. **Add filters**: By type (temples, museums), price range, time availability
4. **Deploy online**: Heroku, Streamlit Cloud, or AWS

### Mid-term:
1. **User profiles**: Save preferences, trip history
2. **Collaborative filtering**: "Users who visited X also liked Y"
3. **Real-time data**: Integrate Google Places API for live ratings
4. **Multi-city trips**: Generate itineraries across multiple cities

### Long-term:
1. **Deep learning**: Use transformers for text descriptions
2. **Reinforcement learning**: Optimize itineraries through user feedback
3. **Mobile app**: React Native or Flutter
4. **Social features**: Share itineraries, follow travelers

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📝 License

This project is open-source and available for educational purposes.

---

## 👤 Author

Built with ❤️ for travelers who want to explore India efficiently.

**Tech Stack**: Python, XGBoost, Pandas, NumPy, Scikit-learn, Streamlit

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

### 🌟 Star this repo if you found it helpful!

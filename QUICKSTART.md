# 🚀 Quick Start Guide

Get started with Travel Maker in under 5 minutes!

---

## ⚡ Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements_clean.txt
```

Or if you prefer using the original requirements:
```bash
pip install -r requirements.txt
```

---

## 🎯 Running the Application

### Option 1: Streamlit Web UI (Recommended)

```bash
streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

### Option 2: Command Line Interface

**Train the model:**
```bash
python src/train.py
```

**Get top places:**
```bash
python src/predict.py
```

**Generate itinerary:**
```bash
python src/itinerary.py
```

---

## 📸 Screenshots & Features

### 🔍 Explore Places Mode
- Search for top-rated places in any Indian city
- View ratings, visit times, and ML scores
- See detailed information like entrance fees and categories

### 📅 Generate Itinerary Mode
- Specify your destination city
- Choose number of days (1-7)
- Set daily visiting hours (4-12 hours)
- Get optimized day-wise schedules

---

## 🎨 Customization

### Change City

Edit `src/predict.py` or `src/itinerary.py`:
```python
city = "Mumbai"  # Change to any city in your dataset
```

### Adjust Itinerary Parameters

In `src/itinerary.py`:
```python
hours_per_day = 10.0  # Increase sightseeing hours
days = 3              # Extend trip duration
```

### Modify Model Parameters

In `src/train.py`:
```python
model = XGBRegressor(
    n_estimators=300,    # More trees = better accuracy (slower)
    max_depth=8,         # Deeper trees = more complex patterns
    learning_rate=0.1,   # Faster learning (may overfit)
)
```

---

## 🐛 Troubleshooting

### Error: "Model not found"
**Solution:** Run training first
```bash
python src/train.py
```

### Error: "Dataset not found"
**Solution:** Ensure `dataset/indian_places.xlsx` exists

### Error: "No places found for city"
**Solution:** Check city name spelling (case-insensitive)

### Error: Streamlit port already in use
**Solution:** Use a different port
```bash
streamlit run app.py --server.port 8502
```

---

## 📚 Next Steps

1. **Read the full README.md** for detailed documentation
2. **Explore the code** in `src/` directory
3. **Add more cities** to your dataset
4. **Customize the UI** in `app.py`
5. **Deploy online** using Streamlit Cloud

---

## 💡 Pro Tips

- **Use requirements_clean.txt** for fresh installations (no conda paths)
- **Cache is enabled** in Streamlit - restart if data changes
- **Model trains in seconds** - feel free to experiment
- **Log scale reviews** helps balance rating vs popularity

---

## 🎓 Learning Resources

- **XGBoost Docs**: https://xgboost.readthedocs.io/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Pandas Docs**: https://pandas.pydata.org/docs/

---

Happy Travels! 🌍✈️

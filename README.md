# 🚀 ZeptoFresh Late Delivery Prediction

## 📌 Overview
This project analyzes delivery data from ZeptoFresh (15-minute delivery startup) to:
- Perform Exploratory Data Analysis (EDA)
- Identify data quality issues
- Engineer meaningful features
- Build a late delivery prediction model

---

## 📊 Dataset Description

The dataset contains ~110,000 orders with features like:

- Delivery metrics: delivery_time_mins, prep_time_mins
- Order details: order_value_Rs, items_count
- External factors: rain_flag, weekend
- Customer info: rating, tenure

---

## ⚠️ Data Quality Issues Identified

| Issue | Type | Fix |
|------|------|-----|
| delivery_time = 0 | Invalid data | Removed rows |
| order_value = 2.95L | Outlier | Capped/removed |
| prep_time < 0 | Data entry error | Replaced with median |
| customer_rating missing/0 | Missing values | Imputed median |

---

## 📈 Key Insights

- Delivery time is **right-skewed**
- Strong correlation:
  - delivery_time ↔ refund (0.74)
  - rain ↔ delay (0.48)
- Tier-1 cities show **bimodal patterns**

---

## ⚙️ Feature Engineering

```python
delivery_speed = rider_distance_km / delivery_time_mins
order_complexity = items_count * prep_time_mins
value_per_item = order_value_Rs / items_count
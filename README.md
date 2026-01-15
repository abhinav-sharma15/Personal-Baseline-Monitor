# 🫀 Personal Baseline Monitor  
**Bayesian Health Monitoring App (Streamlit + Google Sheets)**

A lightweight Streamlit app that learns a **personal baseline** for resting heart rate (RHR), optionally adjusted for **sleep** and **steps**, and generates a **probabilistic daily message** such as:

> “Today’s heart rate is unusually high for *you* (92nd percentile), even after accounting for sleep.”

The app is designed to run on **Streamlit Community Cloud** with **Google Sheets** as persistent storage.

---

## ✨ Features

- 📊 Personalized baseline (not population averages)
- 🧠 Bayesian uncertainty-aware insights
- ➕ Add data via UI form
- ⬆️ Upload historical data via CSV
- 🗂️ Durable persistence using Google Sheets
- 🔄 Re-entering a date updates that day (upsert)
- 🚫 Wellness-style insights (no medical claims)

---

## 🧠 How It Works (High Level)

For each selected day:

1. Uses **only prior days** as evidence  
2. Fits a Bayesian regression model:
   - Baseline RHR
   - Effect of sleep
   - Effect of steps
3. Computes a **predictive distribution** for the day
4. Reports how unusual the observed RHR is

Alert levels:
- 🟢 Normal (< 70%)
- 🟡 Elevated (70–90%)
- 🔴 Unusual (> 90%)

---

## 🗂️ Google Sheets Schema

Sheet tab name: `observations`

Headers (row 1 — must match exactly):

- `date`: YYYY-MM-DD
- `rhr`: Resting heart rate (bpm)
- `sleep_hours`: optional (float)
- `steps`: optional (integer)
- `created_at`: auto-filled by the app (UTC)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Persistence**: Google Sheets (service account)
- **Statistics**: Closed-form Bayesian linear regression
- **Deployment**: Streamlit Community Cloud

---

## 🚀 Local Development

### 1. Clone the repository
```bash
git clone https://github.com/your-username/personal-baseline-monitor.git
cd personal-baseline-monitor



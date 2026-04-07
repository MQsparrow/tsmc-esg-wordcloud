# 📊 ESG Text Mining Dashboard  
TSMC Sustainability Direction Analysis

## 🧠 Project Overview
This project analyzes the language used in the 2024 TSMC Sustainability Report to uncover ESG communication patterns.

We aim to explore:
- What themes dominate ESG communication?
- Are narratives action-oriented or people-centric?
- How language differs across sustainability roles

👉 This project transforms raw ESG text into interactive insights.

---

## 🎯 Objectives
- Extract and preprocess ESG report text
- Apply TF-IDF to identify key terms
- Analyze similarity across sustainability roles
- Visualize results using word clouds and interactive charts
- Build a deployable dashboard

📌 Based on our proposal:  
:contentReference[oaicite:0]{index=0}

---

## 📂 Project Structure

````

your-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── crawler.py
│   ├── preprocess.py
│   ├── feature.py
│   ├── analysis.py
│   ├── visualization.py
│
├── app/
│   └── streamlit_app.py
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── notebooks/
│   └── exploration.ipynb
│
├── requirements.txt
└── README.md

````

---

## ⚙️ Pipeline

````
Crawler → Preprocess → TF-IDF → Similarity → Visualization → Streamlit App
````

---

## 🧩 Methods

### 1. Text Processing
- Tokenization
- Stopword removal
- Cleaning

### 2. Feature Engineering
- TF-IDF vectorization

### 3. Analysis
- Cosine similarity
- Role-based comparison

### 4. Visualization
- Word Cloud
- Frequency charts

---

## 📊 Expected Outputs

- 🌐 Interactive dashboard (Streamlit)
- ☁️ ESG word clouds per sustainability role
- 🔁 Toggle: Action vs People perspective
- 📈 Keyword frequency charts
- 🔗 Cross-role similarity insights

---

## 🚀 How to Run

### 1. Clone repo

````bash
git clone https://github.com/MQsparrow/tsmc-esg-wordcloud.git
cd tsmc-esg-wordcloud
````

### 2. Install dependencies

````bash
pip install -r requirements.txt
````

### 3. Run Streamlit

````bash
streamlit run app/streamlit_app.py
````

---

## 🌍 Deployment

Planned deployment:

* Streamlit Cloud (recommended)
* Share via public URL + QR code

---

## 🔮 Future Work

* Sentiment analysis
* Cross-year ESG comparison
* Cross-company benchmarking
* Topic modeling (LDA / BERTopic)

---

## 👥 Team

* 董苡恩
* 范容瑄
* 高詩怡

---

## 📚 Data Source

* TSMC 2024 Sustainability Report (official document)

---

## 📝 Notes

This project is part of a Text Mining course (2026).
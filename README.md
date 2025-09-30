# TikTok Logistic Regression Project — Predicting Verified Status
# TikTok Logistic Regression Project — Predicting Verified Status

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Kaggle Dataset](https://img.shields.io/badge/Data-Kaggle-blue.svg)

## 🔍 Introduction
With the rapid growth of short-form video platforms like TikTok, millions of videos are uploaded every day. Among these, distinguishing between **factual claims** and **personal opinions** is critical for **content moderation** and **user trust**.

TikTok receives a high volume of user reports on potentially misleading or harmful content. Manually reviewing all reports creates a **backlog** and slows down response time. Earlier analysis showed that **verified users are more likely to post opinions** than claims.

In this project, we build a **logistic regression model** to predict whether a user is **verified** based on video and user features. By understanding which factors are most associated with verified accounts, we can generate insights that support TikTok’s broader goal: **improving claim vs opinion detection and prioritizing moderation efforts**.

---

## 🎯 Project Objectives
- Conduct **Exploratory Data Analysis (EDA)** to check distributions, anomalies, and model assumptions.  
- Handle **missing values, outliers, skewed distributions, and class imbalance**.  
- Encode categorical variables with **One-Hot Encoding**.  
- Build and evaluate a **logistic regression model** to predict verified status.  
- Interpret coefficients and extract **business insights** that can guide TikTok moderation strategies.  

---

## 📊 Dataset
This project uses the **TikTok User Engagement Data** downloaded from Kaggle.  
Dataset link: [TikTok User Engagement Data on Kaggle](https://www.kaggle.com/datasets/yakhyojon/tiktok)

The dataset contains:
- `claim_status` — Type of video content (claim or opinion)  
- `video_id` — Unique identifier (not used as a feature)  
- `video_duration_sec` — Video duration in seconds (5–60 seconds)  
- `video_transcription_text` — Transcript text of the video  
- `verified_status` — Target variable (verified or not verified)  
- `author_ban_status` — Ban status of the author  
- Engagement metrics: `video_view_count`, `video_like_count`, `video_share_count`, `video_download_count`, `video_comment_count`  
- Derived feature: `text_length` (character count of transcript)

---

## 🧪 Workflow
### 1. Exploratory Data Analysis (EDA)
- Checked **missing values** → 298 missing in `claim_status`.  
- Identified **class imbalance** in `verified_status` (≈ 94% not verified, 6% verified).  
- Found **severe skewness** in engagement metrics (likes, shares, downloads, comments).  
- Heatmap showed **high multicollinearity** between `video_view_count` and `video_like_count` (r ≈ 0.86).  

### 2. Data Preprocessing
- Applied **log transformation** (`np.log1p`) to engagement metrics to reduce skewness.  
- Used **One-Hot Encoding** for `claim_status` and `author_ban_status` (`drop='first'`).  
- Addressed imbalance with:  
  - `class_weight='balanced'` in logistic regression, or  
  - Oversampling minority class (`verified`) on the training set.  

### 3. Modeling
- Built logistic regression with `scikit-learn`.  
- Train/test split with stratification on `verified_status`.  
- Evaluated using: Accuracy (with caution due to imbalance), Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC.  

### 4. Insights & Recommendations
- Verified accounts show distinct engagement patterns compared to unverified accounts.  
- Removing `video_like_count` helps reduce multicollinearity (keep views, shares, downloads, comments).  
- **Business impact:** TikTok can prioritize flagged content from verified accounts, especially opinions, for faster moderation.  

---

## 🛠️ Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

Key libraries:
- pandas, numpy — data manipulation  
- matplotlib, seaborn — visualization  
- scikit-learn, imbalanced-learn — modeling & imbalance handling  
- statsmodels — statistical checks  

---

## ▶️ Running the Project
1. Clone the repository  
2. Install requirements  
3. Open the Jupyter Notebook:
```bash
jupyter lab
```
4. Run the notebook: **TikTok project LR.ipynb**

---



## ⚖️ Ethics & Privacy
- The dataset comes from Kaggle and is intended for educational and analytical purposes.  
- When working with social media data, always ensure ethical use, avoid bias amplification, and disclose limitations.  

---

## 📜 License
Released under the MIT License.  

---

## 🙌 Acknowledgements
- Dataset from Kaggle: *TikTok User Engagement Data*.  
- Code and analysis developed by Saman.

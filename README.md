# Hotstar Content Analysis

A data analysis and machine learning project on Hotstar streaming content — covering EDA, hypothesis testing, clustering, and classification.

---

## Project Structure

```
project/
├── data/
│   ├── hotstar.csv            # Raw dataset
│   └── hotstar_cleaned.csv    # Cleaned dataset (auto-generated)
├── src/
│   └── hotstar_analysis.py    # Main analysis script
├── README.md
└── requirements.txt
```

---

## Dataset

| Field | Description |
|---|---|
| `hotstar_id` | Unique content ID |
| `title` | Title of the content |
| `description` | Short synopsis |
| `genre` | Content genre (Drama, Action, Comedy, etc.) |
| `year` | Release year |
| `age_rating` | Age certification (U, U/A 7+, U/A 13+, U/A 16+, A) |
| `running_time` | Duration in minutes |
| `seasons` | Number of seasons (TV shows only) |
| `episodes` | Number of episodes (TV shows only) |
| `type` | `movie` or `tv_show` |

- Total records: ~6,877
- Source: Hotstar platform catalogue

---

## What the Script Does

| Section | Description |
|---|---|
| 1 | Import libraries |
| 2 | Load raw data |
| 3 | Data cleaning — fill missing values, remove duplicates |
| 4 | EDA — genre distribution, type split, runtime, trends |
| 5 | Trend analysis — content growth over years |
| 6 | Normality test — Shapiro-Wilk on running time |
| 7 | Hypothesis testing — T-test, Z-test, Chi-Square |
| 8 | Correlation heatmap |
| 9 | K-Means clustering + PCA visualization |
| 10–12 | Model training — Random Forest & Logistic Regression |
| 13 | Model comparison summary |
| 14 | Outlier detection using Z-score |

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Place the dataset**

Make sure `hotstar.csv` is inside the `data/` folder.

**4. Update the file path**

In `src/hotstar_analysis.py`, update line ~22 to point to your local path:
```python
df = pd.read_csv("data/hotstar.csv")
```

**5. Run the script**
```bash
python src/hotstar_analysis.py
```

---

## Requirements

See `requirements.txt` for all dependencies. Main libraries used: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `statsmodels`.

---

## Results

- **Random Forest** and **Logistic Regression** are compared for classifying content type (movie vs TV show)
- K-Means clustering groups content by running time and release year
- Statistical tests check genre runtime differences and distribution assumptions

# Comprehensive Dataset Trend and Pattern Evaluation

### Dataset: **Airline_Reviews_BA.csv**

## 1. Dataset Overview

- **Rows:** 3862
- **Columns:** 9

### Column Details

- **Id** (numeric) — sample: `2477`
- **Customer Name** (categorical) — sample: `c haddei`
- **Review Head** (text) — sample: `seats are super uncomfortable`
- **Review** (text) — sample: `horrible service from boarding to landing i flew from london to marrakech and was very disappointed with the arrangement for boarding my food is horrible the staff are arrogant to me and the seats are super uncomfortable for a 3h40min flight business class today taking the flight from marrakech to london i do not recommend`
- **Date** (categorical) — sample: `22nd September 2024`
- **Airport Name** (numeric) — sample: `-`
- **Seat Type** (categorical) — sample: `Business Class`
- **Ratings(10)** (numeric) — sample: `1.0`
- **Recommended** (categorical) — sample: `no`

## 2. Sentiment Summary

- **1**: 1646
- **0**: 1404
- **2**: 812

![Sentiment Pie](results/Airline_Reviews_BA/plots/sentiment_pie.png)

## 3. Key Drivers & Correlations

- **A strong relationship was found between Recommended and Ratings(10) using anova_eta2 (effect=0.552, p=0, n=3862). This indicates that Recommended increases with Ratings(10).**
  
![Relation Plot](results/Airline_Reviews_BA/plots/relation_1.png)
- **A strong relationship was found between Recommended and Review using anova_eta2 (effect=0.470, p=0, n=3862). This indicates that Recommended increases with Review.**
  
![Relation Plot](results/Airline_Reviews_BA/plots/relation_2.png)
- **A strong relationship was found between Recommended and Review Head using anova_eta2 (effect=0.310, p=7.1e-314, n=3862). This indicates that Recommended increases with Review Head.**
  
![Relation Plot](results/Airline_Reviews_BA/plots/relation_3.png)
- **A strong relationship was found between Recommended and Avg Sentiment All Textcols using anova_eta2 (effect=0.517, p=0, n=3862). This indicates that Recommended increases with Avg Sentiment All Textcols.**
  
![Relation Plot](results/Airline_Reviews_BA/plots/relation_4.png)

## 4. Summary

This comprehensive analysis of British Airways customer reviews primarily aimed to identify the key drivers influencing customer recommendations. The findings unequivocally demonstrate that customer `Ratings(10)` and the aggregated sentiment across all text columns are the strongest positive predictors of whether a customer will recommend the airline. A detailed look at the sentiment distribution reveals a highly polarized customer experience, with a significant number of both positive and negative reviews, alongside a notable segment of neutral feedback, indicating inconsistent service delivery. Furthermore, the specific content within the full `Review` and the concise `Review Head` also plays a substantial role in shaping recommendations

## 5. Recommendations

1. Conduct Root Cause Analysis for Low Ratings:** Systematically analyze reviews associated with `Ratings(10)` of 1-3 to pinpoint specific, recurring pain points (e.g., flight delays, baggage issues, staff attitude, food quality) and prioritize operational improvements based on frequency and severity.
2. Enh
# Comprehensive Dataset Trend and Pattern Evaluation

### Dataset: **Genshin Impact.csv**

## 1. Dataset Overview

- **Rows:** 22255
- **Columns:** 5

### Column Details

- **reviewId** (text) — sample: `d3821eb7-de38-4697-a4df-0e6b708981ea`
- **userName** (categorical) — sample: `Amrita Das`
- **content** (text) — sample: `good sab`
- **score** (numeric) — sample: `5`
- **at** (datetime) — sample: `2025-06-22 23:56:39`

## 2. Sentiment Summary

- **2**: 12120
- **0**: 5736
- **1**: 4399

![Sentiment Pie](results/Genshin Impact/plots/sentiment_pie.png)

## 3. Key Drivers & Correlations

- **A strong relationship was found between Score and Content using spearman (effect=0.674, p=0, n=22255). This indicates that Score increases with Content.**
  
![Relation Plot](results/Genshin Impact/plots/relation_1.png)

## 4. Summary

This analysis of 22,255 product reviews primarily aims to understand the drivers behind user satisfaction scores. The sentiment analysis of review content reveals a predominantly positive user base, with over half of all reviews expressing positive sentiment (54.46%). However, a substantial quarter of reviews are distinctly negative (25.77%), while nearly 20% remain neutral. A strong positive correlation (Spearman's ρ = 0.674) exists between the sentiment expressed in the review content and the assigned numerical score

## 5. Recommendations

1. Deep Dive into Negative Sentiment:** Conduct a detailed thematic analysis on the 5,736 negative reviews (sentiment `0`) to identify recurring pain points, specific feature requests, or critical bugs. Prioritize addressing the most frequently cited issues.
2. Leverage Positive Feedback for Product Development:** Extract key phrases and features consistently highlighted in the 12,120 positive reviews (sentiment `2`) to understand what users value most. Use these insights to inform future product enhancements and marketing messaging.
3. Monitor Sentiment Shifts Over Time:** Implement continuous sentiment tracking using the `at` timestamp to detect emerging issues (dips in sentiment) or successful improvements (rises in sentiment) promptly, allowing for proactive intervention.
4. Categorize Neutral Reviews:** Investigate the 4,399 neutral reviews (sentiment `1`) to determine if they represent unmet expectations, lack of strong opinion,
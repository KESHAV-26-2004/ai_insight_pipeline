# E-Commerce Product Performance and Customer Purchasing Behavior Report

### Dataset: **amazon_sales_2025_INR.csv**

## 1. Dataset Overview

- **Rows:** 15000
- **Columns:** 14

### Column Details

- **Order_ID** (categorical) — sample: `ORD100000`
- **Date** (datetime) — sample: `2025-01-25`
- **Customer_ID** (categorical) — sample: `CUST2796`
- **Product_Category** (categorical) — sample: `Home & Kitchen`
- **Product_Name** (categorical) — sample: `Cookware Set`
- **Quantity** (numeric) — sample: `2`
- **Unit_Price_INR** (numeric) — sample: `25574.41`
- **Total_Sales_INR** (numeric) — sample: `51148.82`
- **Payment_Method** (categorical) — sample: `Credit Card`
- **Delivery_Status** (categorical) — sample: `Returned`
- **Review_Rating** (numeric) — sample: `1`
- **Review_Text** (categorical) — sample: `Waste of money`
- **State** (categorical) — sample: `Sikkim`
- **Country** (categorical) — sample: `India`

## 2. Sentiment Summary

- **2**: 8014
- **0**: 4638
- **1**: 2348

![Sentiment Pie](results/amazon_sales_2025_INR/plots/sentiment_pie.png)

## 3. Key Drivers & Correlations

- **A strong relationship was found between Review Rating and Review Text using spearman (effect=0.811, p=0, n=15000). This indicates that Review Rating increases with Review Text.**
  
![Relation Plot](results/amazon_sales_2025_INR/plots/relation_1.png)

## 4. Summary

This report provides a comprehensive analysis of E-Commerce product performance and customer purchasing behavior from Amazon sales data in 2025, with a primary focus on maximizing total sales and a secondary on enhancing review ratings. The dataset, comprising 15,000 transactions, details product categories, quantities, pricing, payment methods, delivery status, and customer feedback across various Indian states. Sentiment analysis of customer reviews reveals a predominantly positive landscape, with 8014 positive entries, yet a significant 4638 negative reviews indicate critical areas requiring immediate attention, alongside 2348 neutral comments. A strong positive correlation (Spearman's rho = 0.81) between numerical review ratings and the sentiment expressed in review texts confirms the reliability and consistency of customer feedback

## 5. Recommendations

1. Deep Dive into Negative Reviews:** Systematically analyze `Review_Text` for products with low `Review_Rating` (sentiment 0) to pinpoint specific issues such as product quality, functionality, or misleading descriptions, then prioritize targeted product improvements or clearer communication.
2. Optimize Product Category Performance:** Identify top-performing `Product_Category` and `Product_Name` based on `Total_Sales_INR` and high `Review_
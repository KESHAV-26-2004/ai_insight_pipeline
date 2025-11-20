# Hospital Patient Record Analysis and Healthcare Performance Review

### Dataset: **student_depression_dataset.csv**

## 1. Dataset Overview

- **Rows:** 27901
- **Columns:** 18

### Column Details

- **id** (numeric) — sample: `2`
- **Gender** (categorical) — sample: `Male`
- **Age** (numeric) — sample: `33.0`
- **City** (categorical) — sample: `Visakhapatnam`
- **Profession** (categorical) — sample: `Student`
- **Academic Pressure** (numeric) — sample: `5.0`
- **Work Pressure** (numeric) — sample: `0.0`
- **CGPA** (numeric) — sample: `8.97`
- **Study Satisfaction** (numeric) — sample: `2.0`
- **Job Satisfaction** (numeric) — sample: `0.0`
- **Sleep Duration** (categorical) — sample: `'5-6 hours'`
- **Dietary Habits** (categorical) — sample: `Healthy`
- **Degree** (categorical) — sample: `B.Pharm`
- **Have you ever had suicidal thoughts ?** (categorical) — sample: `Yes`
- **Work/Study Hours** (numeric) — sample: `3.0`
- **Financial Stress** (categorical) — sample: `1.0`
- **Family History of Mental Illness** (categorical) — sample: `No`
- **Depression** (numeric) — sample: `1`

## 2. Sentiment Summary


## 3. Key Drivers & Correlations

- **A strong relationship was found between Depression and Have You Ever Had Suicidal Thoughts ? using spearman (effect=0.546, p=0, n=27901). This indicates that Depression increases with Have You Ever Had Suicidal Thoughts ?.**
  
![Relation Plot](results/student_depression_dataset/plots/relation_1.png)
- **A strong relationship was found between Depression and Academic Pressure using spearman (effect=0.472, p=0, n=27901). This indicates that Depression increases with Academic Pressure.**
  
![Relation Plot](results/student_depression_dataset/plots/relation_2.png)
- **A strong relationship was found between Depression and Financial Stress using spearman (effect=0.363, p=0, n=27901). This indicates that Depression increases with Financial Stress.**
  
![Relation Plot](results/student_depression_dataset/plots/relation_3.png)
- **A moderate relationship was found between Depression and Age using spearman (effect=-0.225, p=0, n=27901). This indicates that Depression decreases with Age.**
  
![Relation Plot](results/student_depression_dataset/plots/relation_4.png)

## 4. Summary

response:
GenerateContentResponse(
    done=True,
    iterator=None,
    result=protos.GenerateContentResponse({
      "candidates": [
        {
          "content": {
            "role": "model"
          },
          "finish_reason": "MAX_TOKENS",
          "index": 0
        }
      ],
      "usage_metadata": {
        "prompt_token_count": 1042,
        "total_token_count": 3089
      },
      "model_version": "gemini-2.5-flash"
    }),
)

## 5. Recommendations

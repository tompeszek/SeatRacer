def render():
    instructions = """
# Rowing Performance Analysis Tool

## Overview
This application analyzes rowing results data to determine the relative performance of each athlete and equipment. It uses statistical models to identify individual contributions to boat performance.

## How to Use This Tool

### Step 1: Prepare Your Data
Begin by uploading and/or editing your rowing data in the **Data** tab. Ensure your data contains race results with consistent athlete identifiers.

### Step 2: Select Analysis Options

#### Model Selection
Choose a model from the sidebar options. Each model uses different statistical approaches to evaluate performance.

#### Historical Analysis
If you check **Run historical analysis**, processing will take longer but will show the progression of performance over time.

#### Data Filters
**Include Shell Classes** allows you to filter specific boat classes if needed for your analysis.

#### Model Weights
Adjust how the model weights different factors:
* **Close Races**: Increase weight for races with small margins
* **Recency Weighting**: Give more importance to recent results
* **Stern Bias**: Weight athletes by position (useful if you believe stern positions have greater impact)

#### Additional Parameters
* **Max Allowed Correlation**: Filters athletes who are too strongly correlated with others (e.g., if two athletes always row together, their individual performances cannot be separated)
* **Ignore coxswains**: When checked, assumes coxswains have minimal impact on crew performance
* **Evaluation Over Time**: Used for historical analysis, provides rolling evaluation of performance

### Step 3: Erg Scores (Optional)
If you've selected a weighted model, you can optionally supply ergometer scores for each athlete to incorporate into the analysis.

### Step 4: Analyze Results
The analysis will automatically run when options are changed. Review the results in the various output tabs.

## Tips for Best Results
* Include a diverse set of boat configurations to help the model separate individual performances
* The more race data you provide, the more accurate the analysis will be
* Use the correlation matrix to identify athletes whose performances cannot be reliably separated
* Historical analysis is most useful when you have data spanning multiple seasons or training periods
"""
    return instructions
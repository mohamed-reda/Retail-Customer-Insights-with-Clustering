# Retail Customer Insights with Clustering

This GitHub repository contains a Jupyter notebook with Python code for customer segmentation and visualization using
KMeans clustering. The code reads transaction data from an Excel file, preprocesses the data, performs KMeans clustering
to segment customers, and visualizes the results using various plots.

## Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Customer Segmentation](#customer-segmentation)
- [Visualization](#visualization)

## Introduction

Customer segmentation is a crucial task in understanding and targeting different customer groups. This repository
provides a Jupyter notebook that demonstrates how to segment customers using KMeans clustering based on their
transaction behavior. Additionally, it includes visualizations to better understand the characteristics of each customer
segment.

## Dependencies

Ensure you have the following Python libraries installed before running the notebook:

- pandas
- seaborn
- matplotlib
- scikit-learn

You can install the dependencies using the following command:

```bash
pip install pandas seaborn matplotlib scikit-learn
```

## Usage

1. Clone the repository:

   ```bash
   git clone {url}
   ```

2. Navigate to the repository directory:

   ```bash
   cd customer-segmentation
   ```

3. Open the Jupyter notebook:

   ```bash
   jupyter notebook customer_segmentation.ipynb
   ```

4. Run the notebook cell by cell to execute the code.

## Customer Segmentation

The notebook reads transaction data from the 'online_retail_II.xlsx' file, preprocesses the data by cleaning and
transforming it, and then generates customer features. KMeans clustering is applied to group customers into distinct
segments based on their behavior.

## Visualization

The notebook includes visualizations to help interpret and understand the customer segments. It provides count plots of
the number of customers in each cluster and box plots to visualize the distribution of features across different
clusters. Additionally, t-SNE (t-distributed stochastic neighbor embedding) is used to create a 2D scatter plot for
visualizing the clusters.

Feel free to explore and customize the notebook to suit your specific use case. If you have any questions or
suggestions, please create an issue in this repository.


---


to run jupyter:

jupyter notebook

(Use Control-C to stop this server)

----
pip install -r requirements.txt

python -m pip install jupyter

---
memory profile:

@memory_profiler.profile

python -m memory_profiler main.py

---

from line_profiler_pycharm import profile

@profile

python -m line_profiler main.py.lprof > results.txt

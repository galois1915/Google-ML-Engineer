## Module 1: Introduction
### Course introduction
This course, "Launching into Machine Learning," provides foundational knowledge in machine learning. Throughout the course, you will learn about improving **data quality, performing exploratory data analysis, building and training ML models using Vertex AI AutoML and BigQuery ML, optimizing and evaluating models using loss functions and performance metrics, and creating repeatable and scalable training, evaluation, and test datasets**. The course is designed to help you understand the terminology used in machine learning and develop practical skills in applying ML techniques.

## Module 2: Get to Know Your Data: Improve Data throught Exploratory Data Analysis
### Introduction
In this module, we look at how to improve the quality of our data and how to explore our data by performing exploratory data analysis. We look at the importance of tidy data in Machine Learning and show how it impacts data quality. For example, missing values can skew our results. You will also learn the importance of exploring your data. Once we have the data tidy, you will then perform exploratory data analysis on the dataset.
Learning Objectives
- Perform exploratory data analysis.
- Categorize missing data strategies
- Improve data quality

### Improve Data Quality
In this course, you will learn about the two phases of machine learning: the training phase and the inference phase. The course emphasizes the importance of data in machine learning projects and provides steps for delivering an ML model to production. These steps include data extraction, data analysis, and data preparation. Data quality is assessed through measures such as **accuracy, timeliness, and completeness**. The course also covers ways to improve data quality, such as resolving **missing values, converting data types, and using one-hot encoding** for categorical features. Overall, the course highlights the significance of data quality in influencing the predictive value of machine learning models.
<p style="text-align: center;">
  <img src="./images/ml-pipeline.png" width="800" />
  <img src="./images/way-quality.png" width="800" />
</p>

### Lab Intro: Improve the quality of your data
This lab focuses on improving data quality by addressing common issues of untidy data. The lab covers the following topics:
1. Missing Values:
   - The lab provides solutions for handling missing attribute values in data.
2. Data Feature Conversion:
   - The lab demonstrates how to convert data feature columns to a date time format.
3. Feature Column Renaming:
   - The lab shows how to rename a feature column in the data.
4. Removing Values:
   - The lab explains how to remove a specific value from a feature column.
5. One-Hot Encodings:
   - The lab covers the creation of one-hot encodings, which are useful for representing categorical data in a machine learning model.
6. Temporal Features Conversions:
   - The lab provides examples of converting temporal features, such as dates and times, into a format suitable for machine learning algorithms.

By addressing these issues, the lab helps prepare data for ingestion by machine learning algorithms, ensuring better data quality and more accurate model predictions.

### Lab Demo: Improve the quality of your data
This section of the course focuses on improving data quality. The main topics covered include resolving missing values, converting data feature columns to a data-in format, and creating one hot encoding of categorical features. The lab exercises involve working with a dataset from the California Open Data Portal that contains information about vehicles by zip code. The necessary libraries for the lab include TensorFlow, Pandas, NumPy, Matplotlib, and Seaborn. The dataset is uploaded from a Google Cloud Storage bucket and processed to create a new dataframe with one hot encoding for categorical features. The final dataset is prepared for training models with 53 columns containing numerical and categorical features. The section concludes by emphasizing the importance of data cleaning and preparation before moving on to creating models.


#### Task 1. Set up your environment
#### Task 2. Launch Vertex AI Notebooks instance
#### Task 3. Clone course repo within your Vertex AI Notebooks instance
https://github.com/GoogleCloudPlatform/training-data-analyst.git
#### Task 4. Improve data quality
- Load the dataset
- Read Dataset into a Pandas DataFrame
- DataFrame Column Data Types
- Summary Statistics 
- Checking for Missing Values
- What Are Our Data Quality Issues?
    1. **Data Quality Issue #1**:  
    > **Missing Values**:
    Each feature column has multiple missing values.  In fact, we have a total of 18 missing values.
    2. **Data Quality Issue #2**: 
    > **Date DataType**:  Date is shown as an "object" datatype and should be a datetime.  In addition, Date is in one column.  Our business requirement is to see the Date parsed out to year, month, and day.  
    3. **Data Quality Issue #3**: 
    > **Model Year**: We are only interested in years greater than 2006, not "<2006".
    4. **Data Quality Issue #4**:  
    > **Categorical Columns**:  The feature column "Light_Duty" is categorical and has a "Yes/No" choice.  We cannot feed values like this into a machine learning model.  In addition, we need to "one-hot encode the remaining "string"/"object" columns.
    5. **Data Quality Issue #5**:  
    > **Temporal Features**:  How do we handle year, month, and day?

### What is Exploratory data analysis
Exploratory Data Analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics using visual methods. EDA involves using graphics and basic sample statistics to get a feeling for what information can be obtained from the dataset. The goal of EDA is to obtain theories that can later be tested in the modeling step. EDA techniques are generally graphical and include **scatter plots, box plots, and histograms**. The three popular data analysis approaches are classical analysis, EDA, and Bayesian analysis. Classical analysis imposes a model on the data, while EDA allows the data to suggest admissible models. Bayesian analysis uses probability statements based on prior data to answer research questions about unknown parameters. Data analysts often mix elements of all three approaches in real-world analysis.
-  uncover underlying structure, 
- extract important variables, 
- detect outliers, 
- and anomalies. 
- Test underlying assumptions
- develop parsimonious models
- determine optimal factor settings
-  look at data for trends

<p style="text-align: center;">
  <img src="./images/eda.png" width="800" />
</p>

### How is EDA used in Machine Learning
This section focuses on how exploratory data analysis (EDA) is used in machine learning. EDA is an approach that allows the data to suggest admissible models that best fit the data, rather than imposing deterministic or probabilistic models on the data. The main goal of EDA is to **understand the data, its structure, outliers, and the models suggested by the data**. 

The section covers two main methods of EDA: univariate analysis and bivariate analysis. 

**Univariate analysis** is the simplest form of analyzing data, where the data has only one variable. It focuses on describing the data, summarizing it, and finding patterns. Categorical data can be analyzed using numerical EDA with the help of pandas' crosstab function and visual EDA using Seaborn's countplot function. Continuous data can be analyzed using pandas' describe function and visualized using boxplots, distribution plots, and kernel density estimation plots (KDE plots) in Python using Matplotlib or Seaborn.

**Bivariate analysis**, on the other hand, involves analyzing the relationship between two sets of values. It helps determine if there is a relationship between variables. Python libraries like Matplotlib and Seaborn can be used to analyze bivariate and multivariate data. Seaborn's conditional plots, such as factor plots and joint plots, are powerful tools for visualizing segmented data and relationships between variables.

<p style="text-align: center;">
  <img src="./images/eda2.png" width="800" />
</p>

### Data analysis and visualization
Exploratory Data Analysis (EDA) is an essential step in the machine learning process. Its purpose is to find insights that will help with data cleaning, preparation, and transformation, which are crucial for building accurate machine learning models. Data analysis and data visualization are used at every step of the machine learning process, including data exploration, data cleaning, model building, and presenting results. Some common visualization techniques used in EDA include **histograms, scatter plots, and heat maps**. The goal of EDA is to gain maximum insight into the dataset, identify outliers or anomalies, and identify the most influential features. There are many ways to explore, analyze, and plot data, so it's important to continue expanding your knowledge in this area.

### Lab introduction: Explore the data using Python and BigQuery
In this course, "Launching into Machine Learning" by Google Cloud, you will learn about exploratory data analysis (EDA) using Python and BigQuery. The course covers various topics such as analyzing pandas data frames, creating Seaborn plots for EDA, writing SQL queries to extract specific fields from BigQuery datasets, and performing EDA in BigQuery. The objective is to help you gain a deeper understanding of your data and improve your machine learning models. Throughout the course, you will have access to course materials and resources to support your learning journey. Let's dive in and explore the world of machine learning!

### Lab: Exploratory data analysis using Python and BigQuery

#### Task 1. Launch Vertex AI Notebooks
#### Task 2. Clone a course repo within your Vertex AI Notebooks instance
```python
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
```
#### Task 3. Exploratory data analysis using Python and BigQuery
- Analyze a Pandas Dataframe
- Create Seaborn plots for Exploratory Data Analysis in Python
- Write a SQL query to pick up specific fields from a BigQuery dataset
- Exploratory Analysis in BigQuery

- Loaded data set
- Inspect the data
- Explore the data
    - heatmap
    - displot
    - scatteplot

In this notebook, we will explore data corresponding to taxi rides in New York City to build a Machine Learning model in support of a fare-estimation tool. The idea is to suggest a likely fare to taxi riders so that they are not surprised, and so that they can protest if the charge is much higher than expected.
- Access and explore a public BigQuery dataset on NYC Taxi Cab rides
- Visualize your dataset using the Seaborn library

```python
%%bigquery
# SQL query to get a fields from dataset which prints the 10 records
SELECT
    FORMAT_TIMESTAMP(
        "%Y-%m-%d %H:%M:%S %Z", pickup_datetime) AS pickup_datetime,
    pickup_longitude, pickup_latitude, dropoff_longitude,
    dropoff_latitude, passenger_count, trip_distance, tolls_amount, 
    fare_amount, total_amount 
# TODO 3
FROM
    `nyc-tlc.yellow.trips`
LIMIT 10
```

### Reading
- [Guide to data Quality Managment](https://www.scnsoft.com/blog/guide-to-data-quality-management)
- [Exploratory Data Analysis with Python](https://www.youtube.com/watch?v=-o3AxdVcUtQ)
- [How to investigate a dataset with Python](https://towardsdatascience.com/hitchhikers-guide-to-exploratory-data-analysis-6e8d896d3f7e)

## Module 3: Machine Learning in Practis
In this module, we will introduce some of the main types of machine learning so that you can accelerate your growth as an ML practitioner.
Learning Objectives
Differentiate between supervised and unsupervised learning
Perform linear regression using Scikit-Learn.
Differentiate between regression and classification problems.


## Module 4: Trainig AutoML Models using Vertex AI
In this module, we will introduce training AutoML Models using Vertex AI.
Learning Objectives
Define automated machine learning
Train a Vertex AI AutoML regression model.
Explain how to evaluate Vertex AI AutoML models

### Machine Learning vs Deep Learning
In this lesson, we explore the concept of automated machine learning and its distinction from statistics and deep learning. We start by understanding that **machine learning begins** with a business requirement or problem that needs to be solved. We then learn about the process of wrangling data, including resolving missing values, converting data formats, renaming columns, and creating one-hot coding features. Exploratory Data Analysis (EDA) is introduced as a technique to visualize and understand the data. 

We learn about different **machine learning frameworks** such as scikit-learn, PyTorch, and TensorFlow, with a focus on scikit-learn, a Python library for machine learning. We also explore the differences between machine learning and **statistics**, including data preparation, hypothesis testing, and the nature of the data.

Finally, we briefly touch on **deep learning** as a subset of machine learning methods. Deep learning is implemented as supervised learning and requires large datasets for higher accuracy. It also involves training on GPUs and offers more control over hyperparameter tuning.

<p style="text-align: center;">
  <img src="./images/mlvssta.png" width="800" />
  <img src="./images/dl.png" width="800" />
</p>

### What is automated machine learning
Automated Machine Learning (AutoML) is the process of applying machine learning to real-world problems in a quicker and more efficient way. **It automates various components of the machine learning workflow, such as data readiness, feature engineering, training and hyperparameter tuning, model serving, explainability and interpretability, and deployment to edge devices**. Vertex AI, a platform by Google Cloud, offers automation for these components and more. Throughout the course series, you will learn about real-world examples of Vertex AI's features, including vizier optimization for hyperparameter tuning, managed datasets, feature store, and more.
<p style="text-align: center;">
  <img src="./images/auto.png" width="800" />
</p>

### AutoML Regression model
This section of the course material focuses on using Vertex AI to train machine learning models without writing code. It starts by explaining the differences between statistics, machine learning, and deep learning. Then, it introduces a use case where a team at XYZ Company wants to deliver their first ML model to production. The team has structured data and wants to predict a credit score. They can use Vertex AI's AutoML, which allows them to load data, generate statistics, and build and train their model without writing code. The team selects the Tabular data type and regression classification as their objective. They upload their dataset and confirm that the data is loaded correctly. They then train the model using AutoML and receive an email when the training is complete. Vertex AI provides a unified platform that eliminates the need for writing code and allows team members to manage various stages of the ML workflow. AutoML is a codeless solution that requires minimal technical effort, while custom training provides more flexibility and control over the training application. AutoML is recommended for users without data science expertise, while custom training requires programming experience. AutoML also saves time by reducing data preparation and development efforts. The choice between AutoML and custom training depends on the machine learning objectives and the level of control required. Both options have limits on managed datasets, but there is no limit on data size for unmanaged datasets. Vertex AI offers a versatile platform for managing datasets, training models, evaluating accuracy, tuning hyperparameters, and deploying models for serving predictions.

<p style="text-align: center;">
  <img src="./images/worflow.png" width="800" />
  <img src="./images/auto-custom.png" width="800" />
  <img src="./images/auto-custom2.png" width="800" />
</p>

### Evaluated AutoML Models
In this lesson, we explore ways to evaluate AutoML models, focusing on model **evaluation for structured or tabular data**. The evaluation metrics provide quantitative measurements of how your model performed on the test set. The data set is split into **training, validation, and testing sets**, with the majority of the data in the training set. The validation set is used to tune the model's hyperparameters, while the test set is used to assess the model's performance on new data. We also discuss various **regression metrics**, such as mean absolute error (MAE), root mean square error (RMSE), and R squared. Additionally, we cover **classification metrics** like precision, recall, and F1 score. **Model feature** attributions are examined to understand the impact of each feature on the model. Finally, we learn about using the model for predictions through **batch prediction and online prediction**.
> Endpoints are the machine learning models that are made available for online prediction requests. You can set up endpoints to handle timely predictions for multiple users or in response to application requests.

<p style="text-align: center;">
  <img src="./images/regression-metrics.png" width="800" />
  <img src="./images/classification-metrics.png" width="800" />
  <img src="./images/predictions.png" width="800" />
</p>

### Reading
- [Training AutoML Models](https://cloud.google.com/vertex-ai/docs/training/training)
- [Train an AutoML model - cloud consile](https://cloud.google.com/vertex-ai/docs/training/automl-console)
- [Train an AutoML model (API)](https://cloud.google.com/vertex-ai/docs/training/automl-api)
- [Optimization objectives for tabular AutoML models](https://cloud.google.com/vertex-ai/docs/training/tabular-opt-obj)
- [Train an AutoML Edge model using the Cloud Console](https://cloud.google.com/vertex-ai/docs/training/automl-edge-console)
- [Train an AutoML Edge model using the Ve ex AI API](https://cloud.google.com/vertex-ai/docs/training/automl-edge-api)
- [Evaluate AutoML Models](https://cloud.google.com/vertex-ai/docs/training/evaluating-automl-models)


## Module 5: BigQuery Machine Learning: Develop ML Models Where Your Data Lives
## Module 6: Optimization
In this module we will walk you through how to optimize your ML models.
Learning Objectives
Discuss how to measure model performance objectively using loss functions.
Explain loss functions as the basis for an algorithm called gradient descent.
Explain how to optimize gradient descent to be as efficient as possible.
Identify performance metrics to make business decisions.
## Module 7: Generalization and sampling
## Module 8: Summary


# Laptop Prices Linear Regression using Regex and Difflib
###### _***By Yehia Hesham***_
### Introduction
In this Project, I try extracting any useful information about each laptop and it's pricing and visualize the findings in a manner that is easy to follow, doing so should help us better understand the factors that affect laptop pricing. After that, building a Regression model that can predict the price of a laptop based on its specificaitons and deploying it to Heroku Cloud.

The reason for choosing this project is personal interest in discovering what factors greatly affect pricing and helping the users understand how much do popular features cost and if they're worth the extra cost/premium.

## Datasets
- Laptop-Prices: Contains the data regarding our laptops' specs and prices
- CPUs/GPUs: Contain Performance metrics for all relevant CPUs/GPUs

## Project Features
- Restructuring the Data into an analysis friendly format
- Analysis on Data with insights and drawing conclusions
- Data Preprocessing for the model
- Model Selection
- Parameter Tuning 
- Deployment on Heroku Cloud

## Libraries used:
- [numpy](https://numpy.org/) 
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/) - Used for building a wide selection of machine learning models and measuring their performance.
- [xgboost](https://xgboost.readthedocs.io/en/stable/) - Used for creating XGradientBoost machine learning models 
- [regex](https://docs.python.org/3/library/re.html) - Used to filter out different string using regex operations and custom string editing. 
- [cdifflib](https://docs.python.org/3/library/difflib.html) - Used for comparing sequences (in this project, strings)
- [Flask](https://flask.palletsprojects.com/en/2.1.x/) - Used to create web application
- [joblib](https://joblib.readthedocs.io/en/latest/) - Used to save models and other variables into .h5 files for deployment
- [gunicorn](https://gunicorn.org/) - HTTP Server that is broadly compatible with various web frameworks.

## Main Folders
- app - Contains the Deployed App files uploaded to Heroku Cloud
- datasets - Datasets used

## Main Files
- data_collecting.ipynb - contains the code used to get the data, and describe some information about the data, business understanding and 10 questions that will be used in Exloratory Data Analysis.
- finalp1.ipynb - Full data preprocessing, EDA, Visualization, Model Selection.
- eda_sql.ipynb - python code that queries the database for answering EDA questions.
- data_db.db - Database of the restructured Data (used in eda_sql.ipynb).
- finalp2.ipynb - Hyperparameter Tuning, building Pipeline and saving Joblib files for Deployment.
- ***Other files either include [.py] scripts or [.h5] files used***

#### Resources
Web Service:     https://laptop-prices-prediction.herokuapp.com/
Tableau Story:   https://public.tableau.com/app/profile/yehia.hesham/viz/Dashboard_Laptop_Prices/Story1
Youtube: https:  https://www.youtube.com/watch?v=UwvBIdh7swo&ab_channel=YehiaHesham


## License
**Free Software**

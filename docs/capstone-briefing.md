Overview
The goal of this project is to leverage state-of-the-art Machine Learning cloud technologies and MLOps practices to develop a robust machine learning solution with real-world impact focused on financial forecasting. In this project, you will work specifically on predicting Gold/USD prices and deploying the model using AWS cloud services. This final project offers an exciting opportunity to explore the intersection of machine learning, cloud deployment, and operational best practices.

You may execute this project as-is or a similar project of your choosing with a focus on cloud-based ML operations, but for evaluation in this bootcamp your project must emphasize deploying and managing models on AWS.

History and Context
Gold has been a globally recognized store of value and investment asset for centuries. It plays a significant role in financial markets as a hedge against inflation and economic uncertainty. The gold price in USD fluctuates based on a variety of factors including geopolitical events, currency valuations, and market demand.

With growing volatility in global economic indicators, predicting gold prices has become increasingly important for investors, traders, and financial analysts. Machine learning models capable of forecasting the price of gold in USD can offer a competitive advantage in decision-making and risk management.

Business Need
Accurate prediction of Gold to USD prices on an hourly or daily basis can help traders optimize buying and selling decisions, financial institutions manage risk, and investment platforms offer automated advisory services. Developing such a predictive model and deploying it effectively on the cloud using MLOps best practices will provide valuable practical experience in end-to-end data science, machine learning, and cloud engineering.

Challenge Outline
Step 1: Connect to an API to Retrieve Gold Price Data
Objective
Connect to an API or data source to fetch historical and real-time Gold/USD price data on an hourly or daily basis, such as Quandl, Alpha Vantage, or Yahoo Finance.

Step 2: Build a Model to Predict Gold/USD Prices
Objective
Develop a machine learning model to predict the Gold/USD price using historical time series data. Possible approaches include LSTM neural networks, ARIMA models, or regression-based methods. Handle noise and volatility inherent in financial data.

Step 3: Deploy the Model Using MLOps Concepts on AWS
Objective
Deploy your trained machine learning model on AWS leveraging MLOps principles and DevOps automation. Attempt to fulfill the following goals:

Goals:
TIER 1

Create a Git repository for the project
Set up a virtual environment for dependency management
Commit and push code to GitHub or similar VCS
TIER 2

Track experiments and model versions with MLflow
Deploy the model on AWS using one of the following:
SageMaker (training and inference endpoints)
EC2 Linux instance running the model
Develop a REST API with Flask or FastAPI to serve predictions
Build a pipeline to automate fetching and preprocessing of gold price data
Implement model monitoring and retraining triggered by performance degradation (dont forget data drift ;D)
TIER 3

Containerize the ML model and all dependencies using Docker for consistent deployment
Optional - Advanced : TIER 4

Automate model retraining and redeployment on performance drop using CI/CD tools
BONUS

Build an alerting system (email, SMS, or Telegram) for significant price changes
Incorporate monitoring and logging via AWS CloudWatch, Prometheus, or Grafana to track usage and performance
Exploratory Topics for Future Work

Implement CI/CD pipelines using GitHub Actions, Jenkins, or CircleCI
Optimize cloud resource usage and cost management via AWS Cost Explorer or similar tools
Apply SHAP or LIME for model interpretability and explainability
Timeline
Day	Tasks
1–2	- Project selection
- Data collection
- Deadline for informing the teaching team of project selection. This includes:
1. Explaining if Option 1 or 2 is chosen.
2. Which data is to be used.
3. A clear high-level overview of the deployed product.
3	Model development
4–6	Interface development
7–8	- Testing
- Evaluation
- Documentation
- Deployment
9	Presentation preparation
10	Project presentation day
Additional Reminder: Check the Evaluation Rubric
Before you begin and throughout the project, please refer to the official Project 2 - ML on Cloud | Predicting Gold/USD Prices Using MLOps on AWS (or similar) Evaluation Rubric, which outlines how points are awarded across areas such as data pipeline quality, AWS deployment, model performance, MLOps structure, documentation, and presentation quality.

We strongly recommend checking it before and during the project, as it will help you prioritize deliverables and align your work with the expected standards.

Additional Notes
Teamwork: Work individually or in groups of no more than 2 people.
Presentation: Tailor your presentation for both technical and non-technical audiences.
Summary
This challenge guides you through the full cycle of data retrieval, predictive modeling, and deployment using MLOps best practices on AWS. You will gain hands-on experience with financial time series data, machine learning techniques, cloud services, and production deployment. Successful completion will build necessary skills to manage real-world end-to-end machine learning projects.

Materials
API Providers for Gold Prices:

Quandl
Alpha Vantage
Yahoo Finance via yfinance Python package
Investing.com
NOTE If you struggle with obtaining the data, please move to the same exercise using any form of currency with the Yahoo Finance API:

Source 1
Source 2
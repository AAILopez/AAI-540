# AAI-540
### Project Overview
This project demonstrates a full end-to-end machine learning workflow on Amazon Web Services (AWS). We began by ingesting our dataset into an AWS S3 bucket to serve as a centralized data store. Using a Jupyter Notebook, we leveraged Pandas for exploratory data analysis (EDA) and feature engineering. The engineered features, along with all other relevant variables, were ingested into AWS Feature Store to enable future collaborators to reuse consistent, versioned data for ongoing or new projects.

Next, we queried our feature store using AWS Athena and prepared the data for model deployment. This preparation included splitting the dataset into training, validation, and test sets, applying one-hot encoding to categorical variables, and standardizing numerical features.

For the modeling phase, each team member was responsible for deploying a different algorithm: Luis deployed a logistic regression model, Andrew implemented XGBoost, and Aaron developed a narrow neural network. The primary objective of this project was not to optimize for the best model performance, but to illustrate the practical steps of an applied Machine Learning Operations (MLOps) pipeline.

After deployment to a production endpoint, we established comprehensive monitoring using Amazon CloudWatch. Our monitoring included bias detection, performance metrics, and infrastructure monitoring (e.g., CPU/GPU utilization, latency), all visualized in a unified CloudWatch dashboard.

Finally, we implemented a CI/CD pipeline with the XGBoost model to showcase how the system can rapidly incorporate updates in response to monitor alerts—such as threshold breaches, data drift, or subpar accuracy—and to facilitate seamless model iteration and redeployment.

Getting Started

To run this project, you will need an independent AWS account with the necessary credentials and permissions to provision resources, deploy models, and configure monitoring.



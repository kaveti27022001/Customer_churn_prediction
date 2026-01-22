# 🔁 Customer Churn Prediction – End-to-End MLOps Project

Welcome to this **end-to-end MLOps project**, designed to demonstrate a **robust, production-grade machine learning pipeline** for **Customer Churn Prediction**.

This project is intentionally crafted to **impress recruiters, MLOps engineers, and hiring managers** by showcasing how real-world ML systems are built, automated, deployed, and maintained using modern **MLOps best practices**.

From raw customer data to a fully deployed prediction service, this repository reflects how ML is done **in industry**, not just in notebooks.

---

## 🌟 Project Highlights

* 📉 **Customer Churn Prediction** on structured tabular data
* 🔁 Fully automated **end-to-end MLOps pipeline**
* 🗄️ **MongoDB Atlas** as cloud data source
* 🧪 Schema-based **data validation & feature engineering**
* 🤖 Model training, evaluation & versioning
* ☁️ **AWS S3** as model registry
* 🚀 Deployment on **AWS EC2**
* 📦 **Dockerized application**
* 🔄 **CI/CD automation** using GitHub Actions

---

## 🧠 Business Problem

Customer churn directly impacts revenue. Predicting churn early enables businesses to:

* Identify high-risk customers
* Take proactive retention actions
* Reduce customer acquisition costs

This project builds a **scalable churn prediction system** while focusing heavily on **MLOps engineering**, reproducibility, and automation.

---

## 📁 Project Setup and Structure

### Step 1: Project Template

Create the project structure by executing:

```bash
python template.py
```

This generates a modular, production-ready folder layout.

---

### Step 2: Package Management (Local Imports)

Local packages are configured using:

* `setup.py`
* `pyproject.toml`

📌 Tip: Refer to `crashcourse.txt` for a quick understanding of Python packaging standards.

---

### Step 3: Virtual Environment & Dependencies

```bash
conda create -n churn python=3.10 -y
conda activate churn
pip install -r requirements.txt
pip list
```

This ensures all required libraries and **local packages** are correctly installed.

---

## 📊 MongoDB Setup and Data Management

### Step 4: MongoDB Atlas Configuration

1. Sign up on **MongoDB Atlas**
2. Create a new project
3. Deploy a **free M0 cluster**
4. Create a DB user (username & password)
5. Allow network access from `0.0.0.0/0`
6. Copy the Python connection string and replace `<password>`

---

### Step 5: Pushing Data to MongoDB

* Create a `notebook/` directory
* Add dataset to the folder
* Create `mongoDB_demo.ipynb`
* Push customer data to MongoDB using Python
* Verify data via **Atlas → Database → Browse Collections**

---

## 📝 Logging, Exception Handling & EDA

### Step 6: Logging and Exception Handling

* Implement centralized logging module
* Add custom exception handling
* Test functionality using `demo.py`

---

### Step 7: Exploratory Data Analysis & Feature Engineering

* Perform EDA
* Handle missing values & outliers
* Engineer features for downstream ML pipeline

---

## 📥 Data Ingestion

### Step 8: Data Ingestion Pipeline

* Configure MongoDB connection in `configuration.mongo_db_connections.py`
* Fetch data using `data_access` layer
* Convert MongoDB key-value documents into Pandas DataFrame
* Define ingestion configs & artifacts in:

  * `entity/config_entity.py`
  * `entity/artifact_entity.py`
* Trigger ingestion via training pipeline

#### Environment Variable Setup

```bash
# Bash
export MONGODB_URL="mongodb+srv://<username>:<password>..."

# PowerShell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>..."
```

---

## 🔍 Data Validation, Transformation & Model Training

### Step 9: Data Validation

* Define schema in `config.schema.yaml`
* Implement validation logic in `utils.main_utils.py`
* Validate:

  * Column names
  * Data types
  * Missing values

---

### Step 10: Data Transformation

* Feature encoding & scaling
* Train-test split
* Transformation logic in `components.data_transformation.py`
* Estimators defined in `entity/estimator.py`

---

### Step 11: Model Training

* Train churn prediction model
* Evaluate performance metrics
* Store trained model artifacts

---

## ☁️ AWS Setup for Model Evaluation & Deployment

### Step 12: AWS Configuration

1. Create IAM user with `AdministratorAccess`
2. Generate access keys
3. Set environment variables

```bash
export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
```

4. Configure constants in `constants.__init__.py`

---

### Step 13: Model Evaluation & S3 Model Registry

* Create S3 bucket: `my-model-mlopsproj` (region: `us-east-1`)
* Compare new model with production model
* Push better-performing model to S3 automatically

---

## 🚀 Prediction Pipeline & Model Deployment

### Step 14: Prediction Pipeline

* Build prediction pipeline
* Configure `app.py` with inference & retraining routes

---

### Step 15: UI Setup

* Add `static/` and `templates/` directories
* Enable web-based interaction

---

## 🔄 CI/CD with Docker, GitHub Actions & AWS

### Step 16: CI/CD Setup

* Create `Dockerfile` and `.dockerignore`
* Configure GitHub Actions workflow
* Add GitHub Secrets:

  * `AWS_ACCESS_KEY_ID`
  * `AWS_SECRET_ACCESS_KEY`
  * `AWS_DEFAULT_REGION`
  * `ECR_REPO`

---

### Step 17: EC2 & ECR Deployment

* Create EC2 Ubuntu instance
* Install Docker
* Configure EC2 as **self-hosted GitHub runner**
* Create ECR repository and push Docker images

---

### Step 18: Final Deployment

* Open port **8000** in EC2 security group
* Access the deployed application:

```
http://<EC2-PUBLIC-IP>:8000
```

---

## 🎯 End-to-End Workflow Summary

```
Data Ingestion
   ↓
Data Validation
   ↓
Data Transformation
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Registry (AWS S3)
   ↓
CI/CD (Docker + GitHub Actions)
   ↓
Production Deployment (EC2)
```

---

## 👨‍💻 Author

**Chinmai Kaveti**
🎓 MS in Data Science
💡 Machine Learning | MLOps | Cloud AI

---

⭐ *If this project helped you or impressed you, don’t forget to star the repository!*

# loan-approval-prediction-decision-tree

<h1 align="center">🏦 Loan Approval Prediction using Decision Tree</h1>

<p align="center">
  A machine learning classification project to predict loan approval status using applicant financial, credit, and demographic features.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge">
  <img src="https://img.shields.io/badge/Matplotlib-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
</p>

---

## 📌 Project Overview

This project uses machine learning to predict whether a loan application will be approved or rejected based on applicant financial history, credit score, income, employment status, debt ratio, loan amount, and other risk-related features.

The main model used in this project is a **Decision Tree Classifier**. The workflow includes data exploration, data cleaning, date conversion, categorical encoding, outlier detection, feature scaling, train-test splitting, model training, and classification evaluation.

---

## 🎯 Business Problem

Loan approval is an important decision for banks and financial institutions. Manual loan screening can be time-consuming and inconsistent.

This project demonstrates how machine learning can be used to support loan approval decision-making by predicting approval status based on applicant profile and financial risk indicators.

---

## 📊 Dataset Information

The dataset contains **20,000 loan application records** and **36 columns** before preprocessing.

After outlier removal, the dataset contains **19,158 records**.

### Target Variable

| Target | Description |
|---|---|
| LoanApproved | Indicates whether the loan application was approved or not |

### Main Features

| Feature | Description |
|---|---|
| ApplicationDate | Date of loan application |
| Age | Applicant age |
| AnnualIncome | Applicant annual income |
| CreditScore | Applicant credit score |
| EmploymentStatus | Employment category |
| EducationLevel | Education level |
| Experience | Work experience |
| LoanAmount | Requested loan amount |
| LoanDuration | Loan duration |
| MaritalStatus | Applicant marital status |
| NumberOfDependents | Number of dependents |
| HomeOwnershipStatus | Home ownership category |
| MonthlyDebtPayments | Monthly debt payments |
| CreditCardUtilizationRate | Credit card usage ratio |
| DebtToIncomeRatio | Debt-to-income ratio |
| BankruptcyHistory | Bankruptcy history indicator |
| LoanPurpose | Purpose of loan |
| PreviousLoanDefaults | Previous loan default indicator |
| PaymentHistory | Payment history score |
| SavingsAccountBalance | Savings account balance |
| CheckingAccountBalance | Checking account balance |
| TotalAssets | Total assets |
| TotalLiabilities | Total liabilities |
| MonthlyIncome | Monthly income |
| NetWorth | Applicant net worth |
| InterestRate | Loan interest rate |
| MonthlyLoanPayment | Monthly loan payment |
| TotalDebtToIncomeRatio | Total debt-to-income ratio |
| RiskScore | Applicant risk score |

---

## 🛠️ Tools and Libraries Used

| Tool / Library | Purpose |
|---|---|
| Python | Machine learning programming |
| Pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, model building, and evaluation |
| Jupyter Notebook | Interactive development environment |

---

## ✅ Project Workflow

- ✅ Imported required Python libraries
- ✅ Loaded loan application dataset
- ✅ Explored dataset using `head()`, `info()`, `describe()`, and `shape`
- ✅ Checked unique values for each column
- ✅ Checked missing values
- ✅ Checked duplicate records
- ✅ Converted `ApplicationDate` into datetime format
- ✅ Created exploratory visualizations
- ✅ Encoded categorical columns using LabelEncoder
- ✅ Created a correlation heatmap
- ✅ Detected outliers using the IQR method
- ✅ Removed outlier records
- ✅ Split data into features and target variable
- ✅ Scaled features using StandardScaler
- ✅ Split dataset into training and testing sets
- ✅ Built a Decision Tree Classifier
- ✅ Evaluated model using accuracy, recall, precision, F1-score, and confusion matrix

---

## 📈 Exploratory Data Analysis

The notebook includes visualizations such as:

- 📊 Age distribution by education level
- 📦 Age by employment status boxplot
- 📊 Loan purpose count plot
- 📈 Credit score vs loan approval line plot
- 📊 Marital status distribution
- 🔴 Risk score vs interest rate joint plot
- 📉 Loan amount KDE distribution
- 🔥 Correlation heatmap
- 📦 Outlier analysis using IQR
- 🧮 Confusion matrix heatmap

---

## 🤖 Model Used

### Decision Tree Classifier

The model was built using:

```python
DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=1
)

🔍 Key Observations
The dataset contains a wide range of applicant financial and demographic features.
No missing values were found in the dataset.
Categorical variables were encoded before model training.
Outliers were detected in selected numerical columns and removed using the IQR method.
The Decision Tree model achieved high performance on the test data.
The model may be strongly influenced by risk-related features such as RiskScore, InterestRate, DebtToIncomeRatio, and CreditScore.
⚠️ Important Note About Model Performance

The model achieved very high accuracy. In real-world financial modelling, such high performance should be checked carefully for possible data leakage.

Before using this model as a serious portfolio project, the following should be reviewed:

Check whether RiskScore, InterestRate, BaseInterestRate, or loan pricing fields were created after the loan approval decision.
Check whether any feature directly depends on LoanApproved.
Run feature importance analysis.
Compare performance after removing possible leakage features.
Use cross-validation for more reliable evaluation.

This project should be presented as an educational machine learning classification project.

💼 Business Use Case

This project can be used as a financial analytics prototype for:

🏦 Loan approval prediction
📊 Credit risk screening
💳 Applicant financial profile analysis
📈 Banking decision-support systems
🧾 Risk-based loan classification
🧑‍💼 Financial machine learning portfolio demonstration


📁 Folder Structure
loan-approval-prediction-decision-tree/
│
├── README.md
├── notebook/
│   └── loan_approval_decision_tree.ipynb
│
├── data/
│   └── Loan.csv
│

└── docs/
    └── model-insights.md
📂 Project Files

The main notebook is available in the notebook/ folder:

notebook/loan_approval_decision_tree.ipynb

The dataset should be stored in the data/ folder:

data/Loan.csv
🎯 What I Learned
✅ Loading and exploring financial datasets
✅ Checking missing values and duplicate records
✅ Converting date columns into datetime format
✅ Creating exploratory data visualizations
✅ Encoding categorical variables using LabelEncoder
✅ Detecting outliers using the IQR method
✅ Removing outlier records
✅ Splitting data into features and target variable
✅ Scaling features using StandardScaler
✅ Building a Decision Tree classification model
✅ Evaluating classification models using accuracy, recall, precision, F1-score, and confusion matrix
✅ Preparing a machine learning notebook for GitHub portfolio



🔮 Future Improvements
🔲 Add feature importance chart
🔲 Check and remove possible data leakage features
🔲 Compare multiple models such as Logistic Regression, Random Forest, XGBoost, KNN, and SVM
🔲 Add cross-validation
🔲 Add ROC-AUC score and ROC curve
🔲 Add hyperparameter tuning using GridSearchCV
🔲 Build a Streamlit loan approval prediction app
🔲 Deploy the model as a small financial risk prediction web app
🔲 Add Power BI dashboard for loan approval insights

👨‍💻 Author

Saran Kumar Krishnan

<p> <img src="https://img.shields.io/badge/GitHub-SaranStiff02-black?style=for-the-badge&logo=github"> <img src="https://img.shields.io/badge/Project-Financial_ML-green?style=for-the-badge"> <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"> </p> ```

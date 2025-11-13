---

ScamScan: AI-Powered Fake Website Detection

Capstone Project – Text, Web, and Social Media Analytics

1. Project Overview

ScamScan is an AI-driven system designed to detect fake, fraudulent, or suspicious e-commerce websites.
The project combines machine learning, NLP-based feature extraction, web scraping, URL analysis, domain intelligence, and FastAPI deployment to deliver real-time scam detection.

This repository includes the complete codebase, datasets, notebooks, documentation, and FastAPI backend for the project.


---

2. Repository Structure

scamscan_capstone_project/
│
├── data/                     # Final dataset and raw data
│   └── db_final.csv
│
├── notebooks/                # Phase 1, 2, and 3 development notebooks
│   ├── phase_1.ipynb
│   ├── phase_2.ipynb
│   └── phase_3.ipynb
│
├── vscode_project/           # FastAPI backend code
│   ├── main.py
│   ├── requirements.txt
│   └── models/
│       ├── scamscan_model.pkl
│       └── tfidf_vectorizer.pkl
│
├── documents/                # PPT, report, PDF
│   ├── ScamScan_Presentation.pdf
│   └── Project_Report.pdf
│
└── README.md                 # Project documentation


---

3. Problem Statement

Online scams and fake shopping websites have increased significantly, resulting in financial loss and user trust issues.
There is a strong need for an automated tool that can evaluate website legitimacy in real time.


---

4. Objectives

1. Identify potential scam websites using machine learning.


2. Extract structured features from URLs, webpage content, and domain metadata.


3. Provide a real-time API for scam prediction.


4. Support e-commerce safety, merchant verification, and fraud prevention.




---

5. Use Cases

1. Detecting fake online shopping sites


2. Preventing payment fraud


3. Analysing suspicious promotions and offers


4. Verifying merchant trustworthiness


5. Scanning URLs before completing online transactions




---

6. Technology Stack

Languages: Python
Libraries: Pandas, NumPy, Scikit-learn, SpaCy, BeautifulSoup, TLDExtract, WHOIS
Machine Learning: TF-IDF, RandomForestClassifier
Deployment: FastAPI, Uvicorn
Tools: Google Colab, VS Code, GitHub


---

7. System Architecture

1. Data collection


2. Feature engineering


3. NLP preprocessing


4. ML model training


5. Backend API deployment


6. Real-time inference




---

8. Project Methodology

The project was built in three distinct phases:

Phase 1: Data collection and preprocessing

Phase 2: Feature engineering and model training

Phase 3: API development and deployment



---

9. Feature Engineering

Extracted 47 features, including:

URL length, digit count, special characters

HTTPS presence

Domain and suffix length

WHOIS domain age

Webpage text processing

Scam keyword frequency

TF-IDF vectorisation



---

10. NLP Techniques Used

Tokenisation

Stopword removal

Lemmatization

Keyword extraction

TF-IDF vectorisation

Named entity detection (SpaCy)



---

11. Why Some NLP Techniques Were Not Used

Although emotion detection, topic modelling, and deep NLP models were explored, they were not included due to:

Limited labelled data

Inconsistency across scam pages

High model complexity without proportional accuracy gain


The selected NLP features delivered a strong balance of performance and interpretability.


---

12. Machine Learning Models Used

Baseline: Logistic Regression

Final Model: Random Forest Classifier (best accuracy, recall, and robustness)



---

13. Model Performance

The final RandomForest model outperformed baselines with strong accuracy and balanced precision-recall scores.
Full evaluation metrics are included in phase_3.ipynb.


---

14. API Deployment

The model is deployed using FastAPI with the following endpoints:

/predict

/payment-check

/scan-promotions

/merchant-verification

/risk-score

/full-analysis



---

15. Challenges

1. Noisy website content


2. Inconsistent WHOIS responses


3. Feature imbalance


4. TF-IDF dimensionality control


5. Deployment dependency issues




---

16. Future Scope

1. Deep learning for webpage content


2. Real-time browser extension


3. URL screenshot-based classification


4. Suspicious transaction analysis


5. Mobile app integration




---

17. Demo Video

Link to demo video:
[Insert Google Drive or YouTube link here]


---

18. How to Run the API

Install dependencies:

pip install -r requirements.txt

Start the API:

uvicorn main:app --reload

Open in browser:

http://127.0.0.1:8000/docs


---

If you want, I can also generate a professional project report PDF or review your GitHub repo before submission.

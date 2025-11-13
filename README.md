ScamScan – AI-Powered Fake Website Detection

Capstone Project – Text, Web & Social Media Analytics

This project builds an intelligent system that detects fake, fraudulent, or suspicious e-commerce websites using a combination of machine learning, NLP-based feature engineering, web scraping, URL analysis, and FastAPI deployment.
The solution identifies scam patterns related to fraudulent stores, fake promotions, phishing pages, and suspicious merchant activity.


---

Project Structure

1. Data

Contains the processed dataset (db_final.csv) with extracted features such as:

URL-based features

Domain intelligence features

Web-content features

Scam keyword scores

Entity counts

TF-IDF vectors



---

2. Notebooks

Includes the full end-to-end workflow:

phase_1.ipynb – Dataset creation, feature extraction, preprocessing

phase_2.ipynb – Model training, evaluation, saving ML pipelines

phase_3.ipynb – API-ready feature extraction + integration testing

scamscan.ipynb – Additional testing notebook (optional)



---

3. VS Code Project (FastAPI Backend)

vscode_project/ contains:

main.py – FastAPI implementation

models/ – ML model (scamscan_model.pkl) and TF-IDF vectorizer

requirements.txt – All project dependencies


Deployed locally using:

uvicorn main:app --reload


---

4. Documents

Includes:

Capstone PPT (PDF format)

Project Report

Demo Video Link


Demo Video (Google Drive):
https://drive.google.com/file/d/1PZJl-I92qiqYHAGwHZBLuZqt23jB5cL8/preview

---

Technology Stack

Python (NumPy, Pandas, Scikit-Learn, SpaCy)

BeautifulSoup for Web Scraping

TLDExtract for URL Parsing

WHOIS for Domain Metadata

FastAPI for API Development

Uvicorn for Hosting

Google Colab for Model Training

VS Code for Backend Development

GitHub for Version Control



---

API Endpoints (FastAPI)

1. /predict – Main Fake Website Detector
2. /payment-check – Detects fake payment pages
3. /scan-promotions – Flags suspicious promotion links
4. /merchant-verification – Uses WHOIS to inspect domain age and legitimacy
5. /risk-score – Assigns a numerical website risk score
6. /full-analysis – Combines all checks into one comprehensive report

Swagger Documentation Auto-Generated at:

http://127.0.0.1:8000/docs


---

How to Run Locally

1. Install dependencies:



pip install -r requirements.txt

2. Run FastAPI server:



uvicorn main:app --reload

3. Open Swagger UI:



http://127.0.0.1:8000/docs


---

Summary

This repository includes the complete dataset, notebooks, feature engineering pipeline, ML models, backend API, and project documentation for the ScamScan Fake Website Detection System.
It demonstrates the full lifecycle of a data-driven security solution using text analytics, web mining, and machine learning.

# ===============================
# ScamScan FastAPI – Main Backend
# ===============================

import re
import joblib
import requests
import tldextract
import datetime
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
import spacy

# -----------------------------
# Load ML Model + Vectorizer
# -----------------------------

model = joblib.load("models/scamscan_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
nlp = spacy.load("en_core_web_sm")   # for entity extraction


# -----------------------------
# Initialize FastAPI App
# -----------------------------

app = FastAPI(
    title="ScamScan API",
    description="AI-powered Fake Website Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================
# FEATURE EXTRACTION: ML MODEL (47D)
# ======================================

def extract_features(url: str):
    feats = {}

    # -----------------------------
    # URL structural features
    # -----------------------------
    feats["url_length"] = len(url)
    feats["num_digits"] = sum(c.isdigit() for c in url)
    feats["num_special_chars"] = sum(not c.isalnum() for c in url)
    feats["has_https"] = 1 if url.startswith("https") else 0

    # Extract domain & suffix
    d = tldextract.extract(url)
    feats["domain_len"] = len(d.domain)
    feats["suffix_len"] = len(d.suffix)

    # -----------------------------
    # Domain age (WHOIS)
    # -----------------------------
    try:
        import whois
        w = whois.whois(d.registered_domain)
        creation = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        feats["domain_age_days"] = (datetime.datetime.now() - creation).days
    except:
        feats["domain_age_days"] = -1

    # -----------------------------
    # Scrape Webpage Text
    # -----------------------------
    try:
        html = requests.get(url, timeout=6).text
        soup = BeautifulSoup(html, "html.parser")
        raw = " ".join([tag.get_text(strip=True) for tag in soup.find_all()])
    except:
        raw = ""

    # Clean text
    clean = re.sub(r"[^a-zA-Z ]", " ", raw).lower()
    clean = re.sub(r"\s+", " ", clean).strip()

    feats["text_length"] = len(clean.split())
    feats["scam_keyword_score"] = sum(w in clean for w in ["offer", "discount", "free", "win", "gift"])
    feats["entity_count"] = len(nlp(clean).ents) if clean else 0

    # -----------------------------
    # TF-IDF (37 Features)
    # -----------------------------
    tfidf_vec = tfidf.transform([clean]).toarray()[0]

    # -----------------------------
    # FINAL 47-D FEATURE VECTOR
    # -----------------------------
    final_vec = np.hstack([
        feats["url_length"],
        feats["num_digits"],
        feats["num_special_chars"],
        feats["has_https"],
        feats["domain_len"],
        feats["suffix_len"],
        feats["domain_age_days"],
        feats["text_length"],
        feats["scam_keyword_score"],
        feats["entity_count"],
        tfidf_vec
    ]).reshape(1, -1)

    return final_vec, feats



# ================================
#           API ROUTES
# ================================

@app.get("/")
def home():
    return {"message": "Welcome to ScamScan Fake Website Detector API"}


# -------------------------------
# 1. Fake Website Detection (ML)
# -------------------------------
@app.get("/predict")
def predict(url: str):
    try:
        X, feat_details = extract_features(url)
        pred = model.predict(X)[0]
        label = "FAKE / SCAM WEBSITE" if pred == 1 else "LEGITIMATE WEBSITE"

        return {
            "url": url,
            "prediction": int(pred),
            "label": label,
            "features_used": feat_details
        }

    except Exception as e:
        return {"error": str(e)}



# -------------------------------------------
# 2. Payment Fraud Prevention
# -------------------------------------------
@app.get("/payment-check")
def payment_check(url: str):
    """Check for unsafe payment patterns (UPI-only, no HTTPS, redirects)."""
    results = {}

    try:
        html = requests.get(url, timeout=6).text.lower()
    except:
        return {"error": "Unable to fetch webpage"}

    results["https_enabled"] = url.startswith("https")

    payment_keywords = ["payment", "checkout", "upi", "wallet", "visa", "mastercard", "rupay"]
    results["payment_keywords_found"] = any(w in html for w in payment_keywords)

    has_upi = "upi" in html
    has_card = any(w in html for w in ["visa", "mastercard", "rupay"])

    results["is_upi_only"] = has_upi and not has_card
    results["uses_whatsapp"] = "whatsapp" in html
    results["uses_telegram"] = "telegram" in html

    if not results["https_enabled"] or results["is_upi_only"] or results["uses_whatsapp"] or results["uses_telegram"]:
        results["payment_risk"] = "HIGH RISK"
    else:
        results["payment_risk"] = "SAFE"

    return results



# -------------------------------------------
# 3. Promotions & Deal Monitoring
# -------------------------------------------
@app.get("/scan-promotions")
def scan_promotions(url: str):
    """Analyze website for suspicious promotional language or unrealistic discounts."""
    try:
        html = requests.get(url, timeout=6).text.lower()
    except:
        return {"error": "Unable to fetch webpage"}

    promo_keywords = ["offer", "sale", "deal", "discount", "free", "limited", "flash", "special"]
    keyword_count = sum(k in html for k in promo_keywords)

    discounts = re.findall(r"(\d{1,3})% ?off", html)
    max_discount = max([int(d) for d in discounts], default=0)

    return {
        "promotion_keywords_detected": keyword_count,
        "maximum_discount_found": max_discount,
        "promotion_risk": "Suspicious" if max_discount >= 70 else "Normal"
    }



# -------------------------------------------
# 4. Merchant Verification
# -------------------------------------------
@app.get("/merchant-verification")
def merchant_verification(url: str):
    """Verify domain age, registrar and legitimacy using WHOIS."""
    ext = tldextract.extract(url)
    domain = ext.registered_domain

    try:
        import whois
        data = whois.whois(domain)
    except:
        return {"error": "WHOIS lookup failed"}

    try:
        creation = data.creation_date[0] if isinstance(data.creation_date, list) else data.creation_date
        age_days = (datetime.datetime.now() - creation).days if creation else -1
    except:
        age_days = -1

    return {
        "domain": domain,
        "domain_age_days": age_days,
        "registrar": data.registrar,
        "country": data.country,
        "verification_status": "Likely Legitimate" if age_days > 365 else "Suspicious"
    }



# -------------------------------------------
# 5. Combined Scam Risk Score (0–100)
# -------------------------------------------
@app.get("/risk-score")
def risk_score(url: str):
    """Compute overall scam score (0–100) using ML + rule-based signals."""
    score = 0

    # ML Prediction
    try:
        X, _ = extract_features(url)
        ml_pred = model.predict(X)[0]
        if ml_pred == 1:
            score += 50
    except:
        ml_pred = -1

    # Payment Risk
    pc = payment_check(url)
    if pc.get("payment_risk") == "HIGH RISK":
        score += 25

    # Promotions Risk
    promo = scan_promotions(url)
    if promo.get("promotion_risk") == "Suspicious":
        score += 15

    # Merchant Verification
    mv = merchant_verification(url)
    if mv.get("domain_age_days", 99999) < 90:
        score += 10

    final_score = min(score, 100)
    label = "SCAM / HIGH RISK" if final_score >= 60 else "LIKELY SAFE"

    return {
        "url": url,
        "risk_score": final_score,
        "label": label,
        "ml_prediction": int(ml_pred),
        "payment_check": pc,
        "promotion_scan": promo,
        "merchant_details": mv
    }

# -------------------------------------------
# 6. Full Analysis Endpoint
# -------------------------------------------
@app.get("/full-analysis")
def full_analysis(url: str):
    """Run ALL ScamScan checks and return a complete analysis report."""

    analysis = {}

    # 1. ML-based Fake Website Detection
    try:
        X, feat_details = extract_features(url)
        ml_pred = model.predict(X)[0]
        ml_label = "FAKE / SCAM WEBSITE" if ml_pred == 1 else "LEGITIMATE WEBSITE"
    except Exception as e:
        ml_pred = -1
        ml_label = "ERROR"
        feat_details = {"error": str(e)}

    analysis["ml_prediction"] = {
        "prediction": int(ml_pred),
        "label": ml_label,
        "features_used": feat_details
    }

    # 2. Payment Fraud Safety Check
    payment = payment_check(url)
    analysis["payment_safety"] = payment

    # 3. Promotions & Suspicious Offers
    promotions = scan_promotions(url)
    analysis["promotions"] = promotions

    # 4. Merchant Verification (WHOIS)
    merchant = merchant_verification(url)
    analysis["merchant_verification"] = merchant

    # 5. Final Scam Risk Score
    risk = risk_score(url)
    analysis["overall_risk_score"] = {
        "risk_score": risk.get("risk_score"),
        "label": risk.get("label")
    }

    # Return full structured report
    return {
        "url_analyzed": url,
        "full_report": analysis
    }


# ===============================
# Run locally:
# uvicorn main:app --reload
# ===============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
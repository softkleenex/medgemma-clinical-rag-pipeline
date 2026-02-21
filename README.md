# 🩺 MedGemma Sentinel

> **Bridging the Gap Between Radiology & Patients with RAG-Enhanced Reasoning**

[![Kaggle](https://img.shields.io/badge/Kaggle-Impact%20Challenge-blue.svg)](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-MedGemma--1.5--4B-green.svg)](https://hf.co/google/medgemma-1.5-4b-it)

MedGemma Sentinel is a dual-agent AI application built for the **Kaggle MedGemma Impact Challenge**. It addresses the medical "Translation Gap" by providing high-precision clinical triage for doctors and empathetic, simplified summaries for patients.

---

## 📺 Project Demo Video
[![MedGemma Sentinel Demo](https://img.youtube.com/vi/r7_SRmbvdk8/0.jpg)](https://youtu.be/r7_SRmbvdk8)
*Click the image above to watch our final submission video on YouTube.*

---

## 🏆 Competition & Project Overview
- **Official Competition:** [MedGemma Impact Challenge Overview](https://www.kaggle.com/competitions/med-gemma-impact-challenge/overview)
- **Goal:** To build human-centered healthcare AI applications using MedGemma and other open-weight models from Google’s Health AI Developer Foundations (HAI-DEF).
- **Core Challenge:** Enhancing clinical workflows and patient outcomes through generative AI that is accurate, safe, and empathetic.
- **Competition Period:** Jan 13, 2026 – Feb 24, 2026
- **Development Period:** Feb 21, 2026 (Intensive 1-day rapid prototyping and deployment)

---

## 🚀 Key Features

- **Sentinel-Clinician (Agent A):** High-precision triage assistant. Identifies pathological findings (e.g., in MRI/CT) and provides reasoning triaged as NORMAL or ABNORMAL.
- **Sentinel-Guide (Agent B):** Empathetic patient translator. Converts dense radiological reports into 7th-grade reading level summaries with actionable advice.
- **RAG-Enhanced Reliability:** Integrated with **MedQuAD** (Medical Question Answering Dataset) via **FAISS** vector search to anchor AI reasoning in verified medical knowledge.
- **Atomic Workflow (v19):** A seamless, single-cell pipeline designed for 100% reliability in cloud execution environments.

## 📊 Measured Impact

Our system quantifies its own impact using the **Flesch-Kincaid Grade Level** score:
- **Clinical Baseline:** Grade 15.3 (Requires Graduate-level education to understand).
- **Sentinel Translation:** **Grade 7.2** (Accessible to the general public).
- **Success Metric:** Achieved a significant reduction in linguistic complexity while maintaining absolute clinical fidelity.

---

## 🛠️ Technical Stack

- **Model:** Google MedGemma 1.5-4B (4-bit QLoRA)
- **RAG Engine:** FAISS + SentenceTransformers (`all-MiniLM-L6-v2`)
- **Metrics:** `textstat`, `evaluate` (ROUGE-L)
- **Deployment:** Kaggle T4 x2 GPU Accelerator

---

## 📁 Repository Structure

```text
medgemma/
├── docs/               # Submission guides and local demo video
├── notebooks/          # Final Atomic Notebook (v19) - The core engine
├── src/                # Modular Python scripts for pipeline automation
├── results/            # Final verified clinical reports and high-res samples
└── requirements.txt    # Project dependencies
```

---

## 🚦 Getting Started

1. Clone this repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Configure `HF_TOKEN` in your environment.
4. Run the end-to-end analysis via `notebooks/kaggle_medgemma_sentinel.ipynb`.

---

## 🤝 Acknowledgments
Built for the MedGemma Impact Challenge. Special thanks to Google Health AI and Kaggle for providing the tools to humanize medical technology.

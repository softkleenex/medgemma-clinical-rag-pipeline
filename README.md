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

## 🧠 Critical Analysis & Lessons Learned

Developing MedGemma Sentinel during an intensive 1-day sprint for the MedGemma Impact Challenge pushed our technical boundaries. Deploying advanced multimodal logic and complex Retrieval-Augmented Generation (RAG) pipelines in constrained, ephemeral cloud environments required significant defensive engineering. 

Below is a critical retrospective of the three major hurdles we encountered and the architectural decisions we implemented to guarantee a robust, fault-tolerant deployment.

### 1. Authentication in Cloud Notebooks: Circumventing `401 Unauthorized` Errors
**The Challenge:** 
During rapid prototyping on Kaggle's accelerator infrastructure, our pipeline was severely bottlenecked by recurring `401 Unauthorized` errors when attempting to pull gated model weights. We diagnosed this as a race condition caused by delayed or inconsistent syncing of Kaggle Secrets, which frequently left the notebook environment without valid Hugging Face credentials during the early execution blocks.

**The Solution:** 
We abandoned relying solely on native secret-management abstractions and engineered a multi-layered, fail-safe authentication mechanism. By explicitly extracting the token via `kaggle_secrets.UserDataClient()`, injecting it directly into `os.environ["HF_TOKEN"]`, and enforcing an explicit, blocking `huggingface_hub.login(token=...)` call prior to any model instantiation, we established a rigid dependency graph. This guaranteed the runtime was fully authenticated before downstream API requests were fired, completely eliminating the 401 exceptions across runtime restarts.

### 2. Multimodal VLM Token Synchronization: Resolving the "0 Image Tokens" Error
**The Challenge:** 
When integrating radiological imaging into the triage workflow, early iterations failed completely. The model suffered from severe hallucinations, accompanied by backend warnings indicating `0 image tokens` were processed. Our initial approach relied on naive string-based prompting—manually prepending `<image>` tags to the text input—which failed to trigger the model's visual projection layers.

**The Solution:** 
We identified a critical misalignment in how modern multimodal processors handle visual-linguistic interleaved data. We refactored our input pipeline, abandoning manual string manipulation in favor of strictly utilizing `processor.apply_chat_template`. By enforcing a structured dictionary format for inputs (e.g., `[{"type": "image"}, {"type": "text", "text": "..."}]`), we ensured the image processor correctly registered, chunked, and embedded the visual features alongside the text tokens. This structural synchronization was the absolute turning point, enabling the model to successfully "see" the scans and perform high-precision pathological triage.

### 3. Dynamic Pathing for RAG: Navigating Unpredictable Input Directories
**The Challenge:** 
To anchor Sentinel's clinical reasoning, we integrated a FAISS-based vector search over the MedQuAD dataset. However, Kaggle's dynamic dataset mounting system makes hardcoded absolute pathing highly fragile. Datasets are frequently mapped to unpredictable folder structures depending on how they are attached to the specific notebook session, causing catastrophic `FileNotFound` errors during the RAG indexing phase.

**The Solution:** 
We decoupled the RAG ingestion pipeline from static paths entirely. We engineered an autonomous, `os.walk`-based dynamic discovery mechanism that recursively scans the `/kaggle/input` tree upon initialization. This system intelligently locates and binds the necessary MedQuAD `.csv` or `.json` files regardless of their parent directory's hash or naming conventions. This defensive strategy dramatically increased the resilience of our RAG engine, ensuring a seamless, "zero-config" execution for anyone forking the notebook.

### Conclusion
This intensive sprint underscored the necessity of defensive programming when transitioning models from theoretical research to practical cloud deployments. The solutions engineered for MedGemma Sentinel—explicit state management, strict token formatting, and dynamic environment adaptation—not only resolved our immediate blockers but established a highly reusable, fault-tolerant blueprint for future open-weight healthcare AI applications.

---

## 🤝 Acknowledgments
Built for the MedGemma Impact Challenge. Special thanks to Google Health AI and Kaggle for providing the tools to humanize medical technology.

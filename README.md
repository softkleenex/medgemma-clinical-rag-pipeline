# 🩺 MedGemma Sentinel (Portfolio Edition: Post-Mortem & Retrospective)

> **Bridging the Translation Gap Between Radiology & Patients with RAG-Enhanced Reasoning**

[![Kaggle](https://img.shields.io/badge/Kaggle-Impact%20Challenge-blue.svg)](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-MedGemma--1.5--4B-green.svg)](https://hf.co/google/medgemma-1.5-4b-it)

*This repository contains the final submission, architecture, and critical retrospective for the Kaggle MedGemma Impact Challenge (concluded Feb 2026).*

---

## 🏅 Competition Results
- **Status:** [Pending Evaluation - Results expected March 2026]
- **Final Rank:** TBD
- **Awards:** TBD

---

## 📺 Project Demo Video
[![MedGemma Sentinel Demo](https://img.youtube.com/vi/r7_SRmbvdk8/0.jpg)](https://youtu.be/r7_SRmbvdk8)
*Watch the end-to-end execution of our v19 Atomic Pipeline.*

---

## 🏆 Competition Context
- **Event:** [Google Health AI: MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge/overview)
- **Goal:** Develop human-centered healthcare AI applications utilizing open-weight models from Google’s Health AI Developer Foundations (HAI-DEF).
- **Scale:** 876 participating teams globally.
- **Timeline:** Jan 13, 2026 – Feb 24, 2026
- **Development Sprint:** Feb 21, 2026 (Intensive 1-day rapid prototyping and cloud deployment)

---

## 🚀 The Solution: A Dual-Agent Architecture

Radiology reports are critical but written in dense jargon (e.g., "periventricular white matter changes"), causing severe patient anxiety and clinical communication bottlenecks. 

**MedGemma Sentinel** solves this using an automated, dual-agent pipeline:
1. **Sentinel-Clinician:** A high-precision Vision-Language Model (VLM) that reads scans, retrieves relevant medical guidelines via **RAG**, and outputs an expert clinical triage report (NORMAL/ABNORMAL).
2. **Sentinel-Guide:** An empathetic NLP translator that converts the clinical findings into a 7th-grade reading level summary, providing reassurance and actionable next steps.

### 📊 Measured Impact
We utilized the **Flesch-Kincaid Grade Level** to quantify our human-centered impact:
- **Clinical Baseline:** Grade 15.3 (Requires Graduate-level education).
- **Sentinel Translation:** **Grade 7.2** (Accessible to the general public).
- **Result:** ~53% reduction in linguistic complexity while maintaining 100% clinical fidelity.

---

## 🧠 Critical Retrospective & Engineering Challenges

Developing MedGemma Sentinel during an intensive sprint pushed our engineering boundaries. Deploying advanced VLM logic and RAG pipelines in constrained, ephemeral cloud environments (Kaggle Notebooks) revealed significant gaps between local prototyping and cloud production. 

Below is a critical analysis of the major hurdles and the architectural pivots required to achieve a 100% reliable execution.

### 1. The Cloud State Illusion: Circumventing `401 Unauthorized` Errors
**The Pitfall:** During rapid prototyping on Kaggle's T4x2 accelerators, our pipeline was severely bottlenecked by recurring `401 Unauthorized` errors when pulling gated model weights, despite having accepted the model license.
**The Root Cause:** Kaggle's `UserSecretsClient` and notebook cell execution suffer from state synchronization delays. Relying on `os.environ["HF_TOKEN"]` across multiple cells created a race condition where the `transformers` library instantiated before the environment was authenticated. Furthermore, disparate sub-libraries expected different token variable names (`HF_TOKEN` vs `HUGGINGFACE_HUB_TOKEN`).
**The Engineering Fix (v19 Atomic):** We engineered a fail-safe, "Atomic" execution block. We consolidated the entire pipeline into a single notebook cell, forcing a rigid, synchronous dependency graph. We injected the token into three different environment variable permutations and enforced an explicit, blocking `huggingface_hub.login()` call prior to any model instantiation. This completely eradicated authentication flakiness.

### 2. Multimodal VLM Token Synchronization: Resolving the "0 Image Tokens" Crash
**The Pitfall:** Early multimodal inference attempts failed catastrophically. The model yielded hallucinations, and the backend threw `Prompt contained 0 image tokens but received 1 images` exceptions.
**The Root Cause:** Modern VLMs like Gemma-3 have highly strict tokenization protocols. Naive string-based prompting (e.g., manually typing `<image> Describe this...`) fails because the text processor does not map the string to the model's internal `image_token_id`.
**The Engineering Fix:** We abandoned manual string manipulation and refactored the pipeline to strictly utilize `processor.apply_chat_template()`. By enforcing a structured dictionary format (`[{"type": "image"}, {"type": "text", "text": "..."}]`), we guaranteed the image processor correctly registered, chunked, and embedded the visual pixel values alongside the text tokens. This structural synchronization unlocked the model's visual reasoning capabilities.

### 3. Ephemeral File Systems: The RAG Pathing Trap
**The Pitfall:** Our FAISS-based RAG engine crashed during cloud execution because it could not locate the MedQuAD dataset (`FileNotFoundError`).
**The Root Cause:** Kaggle dynamically mounts datasets. Hardcoding absolute paths (e.g., `/kaggle/input/medquad/MedQuAD.json`) is an anti-pattern, as directory names and hashes change based on how the dataset was attached to the specific session.
**The Engineering Fix:** We decoupled our data ingestion pipeline from static paths entirely. We engineered an autonomous, `os.walk`-based dynamic discovery mechanism that recursively scans the `/kaggle/input` tree upon initialization. This system intelligently locates and binds the necessary `.csv` or `.json` files using regex heuristics, making the RAG engine resilient to unpredictable cloud mounting behavior.

---

## 📁 Repository Structure

```text
medgemma-clinical-rag-pipeline/
├── docs/               # Submission guides and documentation
├── notebooks/          # The final v19 Atomic Notebook (The Core Engine)
├── src/                # Modular Python scripts (TTS generation, pipeline tests)
├── results/            
│   └── final/          # Final verified MD reports and high-res PNG samples
└── .gitignore          # Strict exclusion rules for weights, envs, and large mp4s
```

## ⚙️ Environment & Setup

- **Hardware Target:** Nvidia T4 x2 (Kaggle Accelerator)
- **Key Libraries:** `transformers` (v5+), `peft` (QLoRA), `bitsandbytes` (4-bit quantization), `faiss-cpu`, `sentence-transformers`, `textstat`.

*(End of Project Retrospective)*

# HQNN-for-Catalyst-Prediction
# 🧠 Hybrid Quantum Neural Network (HQNN) for Catalyst Performance Prediction under Data Consistency Variation

This repository contains all scripts, notebooks, and documentation for the research project **“Enhancing Catalyst Performance Prediction with Hybrid Quantum Neural Networks (HQNN): A Comparative Study on Data Consistency Variation.”**

---

## 🚀 Overview
Catalyst performance prediction is a key challenge in chemical process design due to inconsistent and noisy experimental datasets.  
This project explores the robustness of **Hybrid Quantum Neural Networks (HQNNs)** compared with traditional ML models when trained under varying levels of data consistency.

The implementation integrates **PennyLane**, **Qiskit**, **PyTorch**, and **MLflow** to:
- Simulate catalyst data inconsistencies
- Train and benchmark HQNNs against classical neural networks
- Monitor prediction drift automatically
- Evaluate interpretability using SHAP and PCA

---

## 🧩 Project Objectives
1. Develop a controlled data consistency simulation framework.  
2. Design parameter-efficient HQNNs using variational quantum circuits (VQCs).  
3. Implement automated consistency checks and drift monitoring via MLflow.  
4. Integrate SHAP and PCA for feature attribution and model transparency.

---

## 🗂️ Repository Structure
data/ → Raw and processed catalyst datasets
notebooks/ → Weekly Jupyter notebooks (Week 1–6)
scripts/ → Core HQNN, ANN, and drift monitoring code
results/ → Visuals, metrics, and reports
mlruns/ → MLflow experiment logs (Ui for viewing results)
requirements.txt → Python environment dependencies
setup_hqnn_env.ps1 → PowerShell environment setup script
README.md → Project description and usage
SETUP.md → Project set-up instructions


## Activate Environment and install dependencies

# First create the virtual environment
python -m venv hqnn_env

# Acivate the virtual environment

.\hqnn_env\Scripts\Activate.ps1 

# Install all dependencies
pip install -r requirements.txt
# DP-LLM-Experiments
Comparative analysis of LLM-guided hyper-parameter tuning in differentially private machine learning, including experiments on imbalanced stroke classification and insurance cost regression.

# Research Project 2: LLM-Guided Differential Privacy Experiments

This repository contains all resources, experiments, and analysis related to the application of **Large Language Models (LLMs)** for tuning hyperparameters in **Differentially Private Stochastic Gradient Descent (DP-SGD)**. The experiments focus on enhancing privacy-preserving machine learning models using LLM-driven recommendations.

## Overview

We explore how LLMs like **Gemini Pro 2.5** and **Groq-hosted models** (LLaMA 3, Qwen, Deepseek, and Gemma) respond to refined prompts and guide the selection of `epsilon`, `delta`, and `max_grad_norm` parameters for DP-SGD. The study includes both:

- **Binary classification** with class imbalance (Stroke Prediction)
- **Regression** on a skewed target variable (Insurance Cost Prediction)

## Experiments Conducted

### 1. Stroke Prediction (Classification – Highly Imbalanced)
- Demonstrates LLM-guided improvements in `max_grad_norm` selection
- Highlights failure of default DP parameters under high imbalance
- F1 improvement observed only after prompt refinement

### 2. Insurance Cost Prediction (Regression – Log-Transformed Target)
- Shows LLMs adapting to a regression task
- Negative R² values across the board, suggesting poor model-data fit
- Gemini exhibited better reasoning regarding skewed target distributions

### 3. Groq LLM Comparison
- Evaluates consistency and utility of different Groq models (LLaMA3, Gemma2, Qwen, Deepseek)
- Results show strong alignment in reasoning with varied `epsilon` and `max_grad_norm` suggestions

## Key Learnings

- Default DP-SGD settings fail in class-imbalanced settings without guided tuning
- Prompt engineering significantly improves the quality of LLM-suggested parameters
- Linear models may be insufficient for complex real-world data, but DP noise doesn’t drastically worsen their performance when tuned properly

## Technologies & Tools

- Python, NumPy, Pandas
- Open-source LLM APIs (Gemini Pro 2.5, Groq API)
- Differential Privacy Libraries
- Matplotlib / Seaborn (for visualization)

## Author

**Kashish Bansal**  
Bachelor of Software Engineering (Hons), Deakin University  
Supervised by: Muneeb Ul Hassan

---

*For academic or demonstration purposes only.*


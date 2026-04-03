# 🧬 AMDIS: Microbiome-Aware Drug Dosage RL Environment

[![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-orange.svg)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Package%20Manager-Poetry-60a5fa.svg)](https://python-poetry.org/)

**AMDIS** (Agentic Microbiome-Driven Interaction System) is a high-fidelity Reinforcement Learning environment built on the **OpenEnv** framework. It simulates the complex interplay between drug pharmacokinetics, gut microbiome dynamics, and patient health outcomes.

> [!NOTE]
> Traditional pharmacology often overlooks the gut microbiome's role in drug-response variability. AMDIS provides a platform to train RL agents for personalized dosing strategies that account for these microbial mediators.

---

## Quick Start

### 1. Installation
Ensure you have [Poetry](https://python-poetry.org/) installed.

```bash
git clone <repository-url>
cd AMDIS
poetry install
```

### 2. Run the Environment Server
AMDIS uses FastAPI via OpenEnv to host the environment.

```bash
poetry run uvicorn microbiome.server.app:app
```
*The server will start at `http://localhost:8000` by default.*

### 3. Run a Test Simulation
Execute the local test script to see the dynamics in action:

```bash
poetry run python microbiome/test_local.py
```

---

## The Science Behind AMDIS

The environment is modeled as a **nonlinear dynamical system** using the **Generalized Lotka–Volterra (gLV)** framework.

### Microbial Dynamics
I have simulated species interactions and drug sensitivities:
$$\frac{dx_i}{dt} = x_i \left(r_i + \sum_{j=1}^{N} a_{ij} x_j \right) - b_i D_t x_i$$

### Pharmacology Loop
1.  **Action**: Agent administers a drug dosage.
2.  **Pharmacokinetics**: Drug concentration ($D_t$) evolves with natural clearance.
3.  **Metabolism**: Microbial species ($x_i$) metabolize the drug into active or toxic metabolites ($M_t$).
4.  **Health Outcome**: Patient health ($H_t$) changes based on metabolite concentration and natural disease progression.

---

## 🛠 Project Structure

```text
AMDIS/
├── microbiome/
│   ├── server/
│   │   ├── app.py                     # FastAPI Entry Point
│   │   └── microbiome_environment.py  # Core gLV Physics & Logic
│   ├── client.py                      # OpenEnv-compatible Client
│   ├── models.py                      # Pydantic Action/Observation Schemas
│   └── test_local.py                  # Local Simulation Script
├── pyproject.toml                     # Project Dependencies (Poetry)
└── readme.md                          # You are here!
```

---

## RL Formulation

-   **State ($S_t$)**: $[x_1, \dots, x_N, D_t, M_t, H_t]$ (Microbial abundances, Drug/Metabolite levels, Health marker).
-   **Action ($A_t$)**: Continuous drug dosage $\in [0, D_{max}]$.
-   **Reward ($R_t$)**: Heavily penalizes high disease severity ($H_t$) and drug toxicity.
    $$R_t = -|H_t - H_{target}| - \lambda D_t$$



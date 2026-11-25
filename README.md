# Trustworthy-NLP

**Calibration & selective classification**

![Validation](https://github.com/rohanst14/Trustworthy-NLP/actions/workflows/research.yml/badge.svg)

Local TensorFlow text classification with held-out probability calibration and confidence-based abstention.

## Dashboard

[**Open interactive dashboard**](https://rohanst14.github.io/Trustworthy-NLP/) · [Evaluation protocol](docs/research-protocol.md)

Explore published benchmark metrics and import additional experiment reports in JSON format.

## Installation and execution

```bash
git clone https://github.com/rohanst14/Trustworthy-NLP.git
cd Trustworthy-NLP
python -m venv .venv
# Activate the environment for your operating system.
pip install -e . -r requirements-research.txt
python experiments/calibrate.py
python -m unittest discover -s tests -v
```

## Dataset and methodology

20 Newsgroups; disjoint training, calibration, threshold-selection and test partitions.

Metrics: Accuracy, negative log likelihood, Brier score, expected calibration error and selective risk.

See [evaluation protocol](docs/research-protocol.md) for assumptions and limitations. The experiment writes `results/calibration.json`. Successful benchmark jobs publish measured JSON reports to the repository; the dashboard loads the published report automatically when available. 

## Repository organization

| Path | Purpose |
|---|---|
| src/ | Core algorithms and model interfaces |
| experiments/ | Reproducible experiment entry points |
| tests/ | Automated regression tests |
| docs/ | Dashboard and methodology |
| requirements-research.txt | Experiment dependencies |
| .github/workflows/ | Continuous validation |

Use [GitHub Actions](https://github.com/rohanst14/Trustworthy-NLP/actions) to inspect actual test outcomes.

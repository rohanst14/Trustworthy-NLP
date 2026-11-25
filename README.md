# Trustworthy-NLP

### Selective classification and interpretable token evidence

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) ![Tests](https://github.com/rohanst14/Trustworthy-NLP/actions/workflows/tests.yml/badge.svg) ![Stage](https://img.shields.io/badge/stage-research%20prototype-64748b)

A compact binary text classifier that exposes class probabilities and token-level log odds, and abstains when evidence is insufficient.

## Research extension: TensorFlow reliability evaluation

Train a local TF-IDF neural classifier on 20 Newsgroups. Disjoint fitting, calibration and policy-selection splits prevent test leakage. The experiment measures expected calibration error, Brier score, negative log likelihood and risk–coverage tradeoffs.

```bash
pip install -e . -r requirements-research.txt
python experiments/calibrate.py
python -m unittest discover -s tests -v
```

See [protocol](docs/research-protocol.md) and [research workflow](.github/workflows/research.yml). Full benchmark execution is opt-in via Actions > Calibration validation > Run workflow. No empirical benchmark result is claimed until a run produces an artifact.

## Design (original baseline)

Labeled documents → Laplace-smoothed multinomial Naive Bayes → log-space posterior → confidence threshold → label or abstention. Unknown words are excluded; an input without known tokens abstains.

## Quick start

```bash
git clone https://github.com/rohanst14/Trustworthy-NLP.git
cd Trustworthy-NLP
python -m venv .venv
# Activate .venv for your operating system.
python -m pip install -e .
python examples/demo.py
python -m unittest discover -s tests -v
```

## Usage

```python
from trustworthy_nlp import NaiveBayes, brier
model = NaiveBayes().fit([("normal safe", 0), ("danger alarm", 1)])
print(model.predict("danger alarm", threshold=0.6))
print(brier([0.8, 0.2], [1, 0]))
```

## API contract

| Function | Description |
|---|---|
| `NaiveBayes.fit(rows)` | Fit on text-label pairs with both binary classes represented. |
| `predict(text, threshold=.7)` | Return class-1 probability, confidence, label or None, and per-token log odds. |
| `brier(probabilities, labels)` | Mean squared probability error on aligned nonempty inputs. |

## Repository map

| Path | Role |
|---|---|
| [src/trustworthy_nlp/](src/trustworthy_nlp/) | Classifier and scoring |
| [examples/demo.py](examples/demo.py) | Synthetic demonstration |
| [tests/test_core.py](tests/test_core.py) | Regression tests |
| [docs/methodology.md](docs/methodology.md) | Evaluation and limitations |
| [ROADMAP.md](ROADMAP.md) | Planned extensions |
| [CI workflow](.github/workflows/tests.yml) | Automated test and demo execution |

## Evaluation and limitations

Measure accuracy, coverage, selective risk, Brier score and calibration on a held-out corpus. Choose thresholds on validation data, not test data. Naive Bayes probabilities are not calibrated by default, and abstention is not a guarantee of out-of-distribution detection. Token log odds are model-specific evidence rather than causal explanations. The synthetic demo is not a benchmark.

## Project status

AI-assisted research prototype; see [roadmap](ROADMAP.md).

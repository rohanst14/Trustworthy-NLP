# Methodology

## Model
Multinomial Naive Bayes with Laplace smoothing and empirical class priors. Compute log likelihoods for known tokens, then normalize the two scores with a stable softmax. Confidence is the maximum class probability. Abstain below threshold or when no vocabulary token is present.

## Explanation
Token log odds equal the difference between smoothed class-conditional log probabilities. Repeated tokens contribute repeatedly to prediction; the explanation mapping lists each distinct token once.

## Evaluation protocol
Use disjoint training, validation and test partitions. Report accuracy, coverage (accepted / total), selective risk (incorrect / accepted), Brier score and calibration curves. Record class balance, threshold, sample counts and dataset provenance. The existing tests and demo are synthetic checks, not measured model performance.

## Limitations
Conditional independence and vocabulary assumptions restrict generalization. Uncalibrated confidence cannot certify reliability; human review is needed for high-stakes uses.

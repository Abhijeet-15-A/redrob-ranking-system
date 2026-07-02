# redrob-ranking-system

# RedRob Hackathon - AI Ranking Engine Submission

This repository contains the production code and artifacts for the candidate ranking evaluation pipeline.

## Sandbox Verification
Our end-to-end small-sample verification sandbox is hosted on Google Colab:
👉 [Link to Google Colab Sandbox](https://colab.research.google.com/drive/1kBBY_0wrG92z5OaJp32KcOJRP8Cg9Nvc?usp=sharing)

## Evaluation Pipeline Execution
To reproduce the evaluation pipeline results from the raw candidates file, run the following exact command:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

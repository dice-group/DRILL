# Deep Reinforcement Learning for Refinement Operators in ALC

This open-source project contains the Pytorch implementation of DRILL, training and evaluation scripts. 
To foster further reproducible research and alleviate hardware requirements to reproduce the reported results, we provide pretrained models on all datasets.

# Installation
Create a anaconda virtual environment and install dependencies.
```
git clone https://github.com/dice-group/DRILL
# Create anaconda virtual enviroment
conda create -n drill_env python=3.9
# Active virtual enviroment 
conda activate drill_env
# Install our developed framework. It may take few minutes
pip install -e .
# Test the installation. No error should occur.
python -c "import ontolearn"
```
# Preprocessing 
Unzip knowledge graphs, embeddings, learning problems and pretrained models.
```
unzip KGs.zip
unzip embeddings.zip
unzip pre_trained_agents.zip
unzip LPs.zip
```
# Prepare DL-Learner
Download DL-Learner.
```
# Download DL-Learner
wget --no-check-certificate --content-disposition https://github.com/SmartDataAnalytics/DL-Learner/releases/download/1.4.0/dllearner-1.4.0.zip
unzip dllearner-1.4.0.zip
# Test the DL-learner framework
dllearner-1.4.0/bin/cli dllearner-1.4.0/examples/father.conf
```
## Reproduce experiments
To ease the reproducibility of our experiments, we prove scripts for training and testing.
- ``` sh reproduce_small_benchmark.sh ``` reproduces results on benchmark learning.
- ``` sh reproduce_large_benchmark.sh ``` reproduces results on 370 benchmark learning.
- ``` drill_train.py``` allows to train DRILL on any desired learning problem.

## Interpretation of Classification Reports
```sh
# Responds to the first (index 0) class expression problem
"0": {
      "TargetConcept": "Grandmother",
      "Target": "Grandmother",
      "Prediction": "Grandmother",
      "TopPredictions": [ ["Grandmother","Quality:1.0"],["\u22a4","Quality:0.6666666666666666"],["Sister","Quality:0.46153846153846156"],...],
      "F-measure": 1.0,
      "Accuracy": 1.0,
      "NumClassTested": 5,
      "Runtime": 0.24213624000549316,
      "Positives": [...],
      "Negatives": [...]
   }
```
## Example of a Summary
1. F-measure for OCEL is negative as F-measure is not reported in DL-Learner
2. NumClassTested in ELTL is -1 as number of expression tested is not reported in DL-Learner.
```sh
##### RESULTS on 18 number of learning problems#####
DrillAverage     F-measure:(avg. 0.96 | std. 0.08)      Accuracy:(avg. 0.95 | std. 0.10)                NumClassTested:(avg. 1271.67 | std. 1879.41)            Runtime:(avg.1.15 | std.1.43)
ocel     F-measure:(avg. -0.01 | std. 0.00)     Accuracy:(avg. 0.94 | std. 0.23)                NumClassTested:(avg. 2501.83 | std. 846.80)             Runtime:(avg.6.78 | std.0.45)
celoe    F-measure:(avg. 0.97 | std. 0.06)      Accuracy:(avg. 0.97 | std. 0.08)                NumClassTested:(avg. 569.33 | std. 1180.74)             Runtime:(avg.5.20 | std.0.99)
eltl     F-measure:(avg. 0.96 | std. 0.09)      Accuracy:(avg. 0.95 | std. 0.13)                NumClassTested:(avg. -1.00 | std. 0.00)         Runtime:(avg.4.28 | std.0.66)
```
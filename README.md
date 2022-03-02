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
wget https://github.com/dice-group/Ontolearn/archive/refs/tags/0.4.0.zip
unzip 0.4.0.zip
cd Ontolearn-0.4.0
python -c 'from setuptools import setup; setup()' develop
python -c "import ontolearn"
cd ..
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

# Reproduce experiments
To ease the reproducibility of our experiments, we prove scripts for training and testing.
- ``` sh reproduce_small_benchmark.sh ``` reproduces results on benchmark learning.
- ``` sh reproduce_large_benchmark.sh ``` reproduces results on 370 benchmark learning.
- ``` drill_train.py``` allows to train DRILL on any desired learning problem.

## Supervised Learning, Prior Knowledge Injection and Positive Only Learning

### Supervised Learning
Consider the following json file storing a learning problem.
```sh
{ "problems": { "Aunt": { "positive_examples": [...], "negative_examples": [...] } } }
```
A classification report of DRILL will be stored in a json file as shown below
```sh
{
   "0": {
      "TargetConcept": "Aunt",
      "Target": "Aunt",
      "Prediction": "Female",
      "TopPredictions": [["Female","Quality:0.804"],["\u00acMale","Quality:0.804"], ... ],
      "F-measure": 0.804,
      "Accuracy": 0.756,
      "NumClassTested": 6117,
      "Runtime": 3.53,
      "positive_examples": [...],
      "negative_examples": [...]
   },
```
### Supervised Learning with Prior Knowledge Injection
Currently, we are exploring the idea of injecting prior knowledge into DRILL.
```sh
{ "problems": { "Aunt": { "positive_examples": [...], "negative_examples": [...],"ignore_concepts": ["Male","Father","Son","Brother","Grandfather","Grandson"] } } }
```
A class expression report will be obtained while ignoring any expression related to "ignore_concepts"
### From Supervised Learning to Positive Only Learning
Currently, we are exploring the idea of applying a pretrained DRILL that is trained for Supervised Learning in positive only learning.
```sh
{ "problems": { "Aunt": { "positive_examples": [...], "negative_examples": [] # Empty list} } }
```
## How to cite
```
@article{demir2021drill,
  title={DRILL--Deep Reinforcement Learning for Refinement Operators in $$\backslash$mathcal $\{$ALC$\}$ $},
  author={Demir, Caglar and Ngomo, Axel-Cyrille Ngonga},
  journal={arXiv preprint arXiv:2106.15373},
  year={2021}
}
```

For any further questions or suggestions, please contact:  ```caglar.demir@upb.de``` / ```caglardemir8@gmail.com```
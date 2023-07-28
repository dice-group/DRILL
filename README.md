# Neuro-Symbolic-Class-Expression-Learning

This open-source project contains the Pytorch implementation of DRILL, training and evaluation scripts. 
To foster further reproducible research and alleviate hardware requirements to reproduce the reported results, we provide pretrained models on all datasets.

# Installation
Create a anaconda virtual environment and install dependencies.
```
git clone https://github.com/dice-group/DRILL
# Create anaconda virtual enviroment
conda create -n drill python=3.10 && conda activate drill 

# Dependencies
pip install dicee
pip3 install parsimonious
pip3 install rdflib
```
# Preprocessing 
Unzip knowledge graphs, embeddings, learning problems and pretrained models.
```
unzip KGs.zip
unzip embeddings.zip
unzip pre_trained_agents.zip
unzip LPs.zip
```

# Training

## Knowledge Graph Embeddings
#### Install dice-embeddings framework
```
git clone https://github.com/dice-group/dice-embeddings.git
pip install -r dice-embeddings/requirements.txt
mkdir -p dice-embeddings/KGs/Biopax
```
Convert an OWL knowledge base into ntriples to create training dataset for KGE.
```python
import rdflib
g = rdflib.Graph()
g.parse("KGs/Biopax/biopax.owl")
g.serialize("dice-embeddings/KGs/Biopax/train.txt", format="nt")
```
#### Compute Embeddings
Executing the following command results in creating a folder (KGE_Embeddings) containing all necessary information about the KGE process.
```
python dice-embeddings/main.py --path_dataset_folder "dice-embeddings/KGs/Biopax" --storage_path "KGE_Embeddings" --model "ConEx"
```
## Train DRILL
To train DRILL, we need to provide the path of a knowledgebase (KGs/Biopax/biopax.owl) and embeddings
```
python drill_train.py --path_knowledge_base "KGs/Biopax/biopax.owl" --path_knowledge_base_embeddings "KGE_Embeddings/2022-05-13 11:02:53.276242/ConEx_entity_embeddings.csv" --num_episode 2 --min_num_concepts 2 --num_of_randomly_created_problems_per_concept 1 --relearn_ratio 5 --use_illustrations False
```

### Run Endpoint
A crude workaround for running endpoint. I would suggest to create your own endpoint :)
```
git clone https://github.com/dice-group/Ontolearn.git
cd Ontolearn
git checkout bf2f94f56bf4508b53a540b5e580a59d73689ccb 
pip install -e .
cd ..
python Ontolearn/examples/simple_drill_endpoint.py --path_knowledge_base 'KGs/Biopax/biopax.owl' --path_knowledge_base_embeddings 'KGE_Embeddings/2022-05-13 11:02:53.276242/ConEx_entity_embeddings.csv' --pretrained_drill_avg_path 'Log/20220513_110334_403223/DrillHeuristic_averaging.pth'
```
### Send a Request
```
jq '
   .problems
     ."((pathwayStep ⊓ (∀INTERACTION-TYPE.Thing)) ⊔ (sequenceInterval ⊓ (∀ID-VERSION.Thing)))"
   | {
      "positives": .positive_examples,
      "negatives": .negative_examples
     }' LPs/Biopax/lp.json         | curl -d@- http://0.0.0.0:9080/concept_learning

```

# Reproduce experiments
### Prepare DL-Learner
Download DL-Learner.
```
# Download DL-Learner
wget --no-check-certificate --content-disposition https://github.com/SmartDataAnalytics/DL-Learner/releases/download/1.4.0/dllearner-1.4.0.zip
unzip dllearner-1.4.0.zip
# Test the DL-learner framework
dllearner-1.4.0/bin/cli dllearner-1.4.0/examples/father.conf
```

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

# from ontolearn import KnowledgeBase, LearningProblemGenerator, DrillAverage, DrillProbabilistic
# from ontolearn.util import sanity_checking_args
from argparse import ArgumentParser
import os
import json
from reasoner import SPARQLCWR
import random
import torch
import numpy as np
from drill import DrillAverage
from metrics import F1
from typing import Iterable,Set, Tuple
random_seed = 1
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


class LearningProblemGenerator:
    def __init__(self, reasoner):
        self.reasoner = reasoner

    def generate(self) -> Iterable[Tuple[Set[str], Set[str], Set[str], str]]:
        """ Generate learning problems """
        # Convert into generator later on
        result=[]
        for c in self.reasoner.get_named_concepts():
            individuals = self.reasoner.retrieve(c)
            assert isinstance(individuals, set)
            if len(individuals) > 3:
                pos = set(random.sample(individuals, 3))
                neg = set(random.sample(self.reasoner.individuals, 3))
                result.append((pos, neg, individuals, c.iri))
        return result


# from owlapy.parser import DLSyntaxParser
# from owlapy.owl2sparql.converter import Owl2SparqlConverter
# parser = DLSyntaxParser("http://www.benchmark.org/family#")
# converter = Owl2SparqlConverter()
# print(converter.as_query("?var", parser.parse_expression('≥ 2 hasChild.Mother'), False))

class Trainer:
    def __init__(self, args):
        self.args = args

        assert self.args.endpoint

    def save_config(self, path):
        with open(path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(self.args)
            json.dump(temp, file_descriptor)

    def start(self):
        # 1. Parse KG.
        reasoner = SPARQLCWR(url=self.args.endpoint, name='Fuseki')

        # 2. Generate Learning Problems.
        lp = LearningProblemGenerator(reasoner=reasoner)

        drill = DrillAverage(reasoner=reasoner, pretrained_model_path=self.args.pretrained_drill_avg_path,
                             quality_func=F1(),
                             drill_first_out_channels=self.args.drill_first_out_channels,
                             path_of_embeddings=self.args.path_knowledge_base_embeddings,
                             gamma=self.args.gamma,
                             num_of_sequential_actions=self.args.num_of_sequential_actions,
                             num_episode=self.args.num_episode,
                             max_len_replay_memory=self.args.max_len_replay_memory,
                             epsilon_decay=self.args.epsilon_decay,
                             num_episodes_per_replay=self.args.num_episodes_per_replay,
                             num_epochs_per_replay=self.args.num_epochs_per_replay,
                             relearn_ratio=self.args.relearn_ratio,
                             use_target_net=self.args.use_target_net,
                             batch_size=self.args.batch_size, learning_rate=self.args.learning_rate,
                             use_illustrations=self.args.use_illustrations,
                             verbose=self.args.verbose,
                             num_workers=self.args.num_workers)
        drill.train(lp.generate())
        print('Completed.')


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--endpoint", type=str, default='http://localhost:3030/mutagenesis/')
    parser.add_argument("--path_knowledge_base_embeddings", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpus used during batching')

    # Concept Generation Related
    parser.add_argument("--min_num_concepts", type=int, default=2)
    parser.add_argument("--min_length", type=int, default=4, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=5, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_ratio_per_concept", type=float, default=.01)  # %1
    parser.add_argument("--max_num_instances_ratio_per_concept", type=float, default=.60)  # %30
    parser.add_argument("--num_of_randomly_created_problems_per_concept", type=int, default=2)

    # DQL related
    parser.add_argument("--gamma", type=float, default=.99, help='The discounting rate')
    parser.add_argument("--num_episode", type=int, default=100, help='Number of trajectories created for a given lp.')
    parser.add_argument("--epsilon_decay", type=float, default=.01, help='Epsilon greedy trade off per epoch')
    parser.add_argument("--max_len_replay_memory", type=int, default=1024,
                        help='Maximum size of the experience replay')
    parser.add_argument("--num_epochs_per_replay", type=int, default=3,
                        help='Number of epochs on experience replay memory')
    parser.add_argument("--num_episodes_per_replay", type=int, default=10, help='Number of episodes per repay')
    parser.add_argument('--num_of_sequential_actions', type=int, default=2, help='Length of the trajectory.')
    parser.add_argument('--relearn_ratio', type=int, default=2, help='# of times lps are reused.')
    parser.add_argument('--use_illustrations', default=True, type=eval, choices=[True, False])
    parser.add_argument('--use_target_net', default=False, type=eval, choices=[True, False])

    # The next two params shows the flexibility of our framework as agents can be continuously trained
    parser.add_argument('--pretrained_drill_avg_path', type=str,
                        default='', help='Provide a path of .pth file')
    # NN related
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=int, default=.01)
    parser.add_argument("--drill_first_out_channels", type=int, default=32)

    # Concept Learning Testing
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3, help='Max. runtime during testing')

    trainer = Trainer(parser.parse_args())
    trainer.start()

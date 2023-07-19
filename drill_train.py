from ontolearn import KnowledgeBase, LearningProblemGenerator, DrillAverage, DrillProbabilistic
from ontolearn.util import sanity_checking_args
from argparse import ArgumentParser
import os
import json

import random
import torch
import numpy as np

random_seed = 1
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


class Trainer:
    def __init__(self, args):
        sanity_checking_args(args)
        self.args = args

    def save_config(self, path):
        with open(path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(self.args)
            json.dump(temp, file_descriptor)

    def start(self):
        # 1. Parse KG.
        if self.args.endpoint is not None:
            # @ TODO Establish a connection to a concept retriever
            # @ TODO SPARQL endpoint, ontolearn or a KGE
            raise NotImplementedError('A binding is needed')
        kb = KnowledgeBase(self.args.path_knowledge_base)
        min_num_instances = self.args.min_num_instances_ratio_per_concept * len(kb.individuals)
        max_num_instances = self.args.max_num_instances_ratio_per_concept * len(kb.individuals)
        # 2. Generate Learning Problems.
        lp = LearningProblemGenerator(knowledge_base=kb,
                                      min_length=self.args.min_length,
                                      max_length=self.args.max_length,
                                      min_num_instances=min_num_instances,
                                      max_num_instances=max_num_instances)
        balanced_examples = lp.get_balanced_n_samples_per_examples(
            n=self.args.num_of_randomly_created_problems_per_concept,
            min_num_problems=self.args.min_num_concepts,
            num_diff_runs=self.args.min_num_concepts // 2)

        drill = DrillAverage(pretrained_model_path=self.args.pretrained_drill_avg_path,
                             knowledge_base=kb,
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
        self.save_config(drill.storage_path)
        drill.train(balanced_examples)
        print('Completed.')


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base",
                        default='KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--endpoint",
                        default='?')
    parser.add_argument("--path_knowledge_base_embeddings",
                        default='embeddings/ConEx_Family/ConEx_entity_embeddings.csv')
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

from abc import ABCMeta
from abstracts import AbstractScorer

import time
import json
import pandas as pd
import numpy as np
import functools
from typing import Set, Tuple
import torch
from torch import nn
from torch.functional import F
from torch.nn.init import xavier_normal_


class DrillAverage:

    def __init__(self, reasoner,
                 path_of_embeddings=None,
                 drill_first_out_channels=32,
                 refinement_operator=None, quality_func=None, gamma=None,
                 pretrained_model_path=None, iter_bound=None, max_num_of_concepts_tested=None, verbose=None,
                 terminate_on_goal=True, ignored_concepts=None,
                 max_len_replay_memory=None, batch_size=None, epsilon_decay=None,
                 num_epochs_per_replay=None, num_episodes_per_replay=None, learning_rate=None,
                 relearn_ratio=None, use_illustrations=None,
                 use_target_net=False,
                 max_runtime=None, num_of_sequential_actions=None, num_episode=None, num_workers=32):

        self.sample_size = 1
        self.embedding_dim = 32
        self.drill_first_out_channels = 3
        self.learning_rate = .1
        arg_net = {'input_shape': (4 * self.sample_size, self.embedding_dim),
                   'first_out_channels': self.drill_first_out_channels, 'num_output': 1, 'kernel_size': 3}

        self.heuristic_func = DrillHeuristic(mode='averaging', model_args=arg_net)
        self.optimizer = torch.optim.Adam(self.heuristic_func.net.parameters(), lr=self.learning_rate)
        if pretrained_model_path:
            try:
                m = torch.load(pretrained_model_path, torch.device('cpu'))
                self.heuristic_func.net.load_state_dict(m)
                for parameter in self.heuristic_func.net.parameters():
                    parameter.requires_grad = False
                self.heuristic_func.net.eval()
                print('DRILL is loaded.')
            except FileNotFoundError:
                raise FileNotFoundError(f'Could not find a pretrained model under {pretrained_model_path}.')

        print('Number of parameters: ', sum([p.numel() for p in self.heuristic_func.net.parameters()]))

    def rl_learning_loop(self,pos,neg):
        # 2. Obtain embeddings of positive and negative examples.
        if False:
            emb_pos = torch.tensor(self.instance_embeddings.loc[list(pos)].values, dtype=torch.float32)
            emb_neg = torch.tensor(self.instance_embeddings.loc[list(neg)].values, dtype=torch.float32)

            emb_pos = torch.mean(emb_pos, dim=0)
            emb_pos = emb_pos.view(1, 1, emb_pos.shape[0])
            emb_neg = torch.mean(emb_neg, dim=0)
            emb_neg = emb_neg.view(1, 1, emb_neg.shape[0])
        else:
            emb_pos=torch.rand(5)
            emb_neg=torch.rand(5)


        #root = self.rho.getNode(self.start_class, root=True)
        #self.assign_embeddings(root)
        #sum_of_rewards_per_actions = []

        print('RL agent starts to interact with the environment. Trajectories will be summarized.')

        exit(1)
        for th in range(1, self.num_episode + 1):



            sequence_of_states, rewards = self.sequence_of_actions(root)
            if th % log_every_n_episodes == 1:
                self.describe_single_rl_loop(th,
                                             sequence_of_states,
                                             rewards=rewards)

            # (3.2) Form experiences for Experience Replay
            self.form_experiences(sequence_of_states, rewards)
            sum_of_rewards_per_actions.append(sum(rewards))

            # (3.3) Experience Replay
            if th % self.num_episodes_per_replay == 0:
                self.learn_from_replay_memory()

            if self.target_net and th % (self.num_episodes_per_replay * 3) == 0:
                self.target_net.load_state_dict(self.heuristic_func.net.state_dict())

            # (3.4) Epsilon greedy => Exploration Exploitation
            self.epsilon -= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                break
        return sum_of_rewards_per_actions

    def train(self, learning_problems: iter):
        for pos, neg, true_pos, str_concept in learning_problems:
            for _ in range(2): # self.relearn_ratio
                sum_of_rewards_per_actions = self.rl_learning_loop(pos=pos,neg=neg)
            exit(1)

        exit(1)

    def fit(self, pos: Set[str], neg: Set[str], ignore: Set[str] = None):
        """
        Find hypotheses that explain pos and neg.
        """
        # 1. Set default rl state
        self.default_state_rl()
        # 2. Initialize learning problem
        if len(ignore) == 0:
            ignore = None
        self.initialize_learning_problem(pos=pos, neg=neg, all_instances=self.kb.thing.instances, ignore=ignore)
        # 3. Prepare embeddings of positive and negative examples
        self.emb_pos, self.emb_neg = self.represent_examples(pos=pos, neg=neg)

        # 4. Set start time for the first criterion for termination
        self.start_time = time.time()
        # 5. If w
        if len(self.concepts_to_ignore) > 0:
            for i in self.concepts_to_ignore:
                if self.verbose > 1:
                    print(f'States includes {i} will be ignored')
                self.rho.remove_from_top_refinements(i)
        else:
            self.rho.compute_top_refinements()
        # 5. Iterate until the second criterion is satisfied.
        for i in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(i)
            next_possible_states = []
            for ref in self.apply_rho(most_promising):
                # Instance retrieval.
                ref.concept.instances = self.kb.instance_retrieval(ref.concept)
                if len(ref.concept.instances):
                    # Compute quality
                    self.search_tree.quality_func.apply(ref)
                    if ref.quality == 0:
                        continue
                    next_possible_states.append(ref)
                    if ref.quality == 1:
                        break
            try:
                assert len(next_possible_states) > 0
            except AssertionError:
                print(f'DEAD END at {most_promising}')
                raise
            predicted_Q_values = self.predict_Q(current_state=most_promising, next_states=next_possible_states)
            self.goal_found = self.update_search(next_possible_states, predicted_Q_values)
            if self.goal_found:
                if self.terminate_on_goal:
                    return self.terminate()
            if time.time() - self.start_time > self.max_runtime:
                return self.terminate()
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()
        return self.terminate()

    def represent_examples(self, *, pos: Set[str], neg: Set[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Represent E+ and E- by using embeddings of individuals.
        Here, we take the average of embeddings of individuals.
        @param pos:
        @param neg:
        @return:
        """
        assert isinstance(pos, set) and isinstance(neg, set)

        emb_pos = torch.tensor(self.instance_embeddings.loc[list(pos)].values,
                               dtype=torch.float32)
        emb_neg = torch.tensor(self.instance_embeddings.loc[list(neg)].values,
                               dtype=torch.float32)
        assert emb_pos.shape[0] == len(pos)
        assert emb_neg.shape[0] == len(neg)

        # Take the mean of embeddings.
        emb_pos = torch.mean(emb_pos, dim=0)
        emb_pos = emb_pos.view(1, 1, emb_pos.shape[0])
        emb_neg = torch.mean(emb_neg, dim=0)
        emb_neg = emb_neg.view(1, 1, emb_neg.shape[0])
        return emb_pos, emb_neg

    def init_training(self, pos_uri: Set[str], neg_uri: Set[str]) -> None:
        """

        @param pos_uri: A set of positive examples where each example corresponds to a string representation of an individual/instance.
        @param neg_uri: A set of negative examples where each example corresponds to a string representation of an individual/instance.
        @return:
        """
        # 1.
        self.reward_func.pos = pos_uri
        self.reward_func.neg = neg_uri

        # 2. Obtain embeddings of positive and negative examples.
        self.emb_pos = torch.tensor(self.instance_embeddings.loc[list(pos_uri)].values, dtype=torch.float32)
        self.emb_neg = torch.tensor(self.instance_embeddings.loc[list(neg_uri)].values, dtype=torch.float32)

        # (3) Take the mean of positive and negative examples and reshape it into (1,1,embedding_dim) for mini batching.
        self.emb_pos = torch.mean(self.emb_pos, dim=0)
        self.emb_pos = self.emb_pos.view(1, 1, self.emb_pos.shape[0])
        self.emb_neg = torch.mean(self.emb_neg, dim=0)
        self.emb_neg = self.emb_neg.view(1, 1, self.emb_neg.shape[0])
        # Sanity checking
        if torch.isnan(self.emb_pos).any() or torch.isinf(self.emb_pos).any():
            print(string_balanced_pos)
            raise ValueError('invalid value detected in E+,\n{0}'.format(self.emb_pos))
        if torch.isnan(self.emb_neg).any() or torch.isinf(self.emb_neg).any():
            raise ValueError('invalid value detected in E-,\n{0}'.format(self.emb_neg))

        # Default exploration exploitation tradeoff.
        self.epsilon = 1

    def terminate_training(self):
        self.save_weights()
        # json serialize
        with open(self.storage_path + '/seen_lp.json', 'w') as file_descriptor:
            json.dump(self.seen_examples, file_descriptor, indent=3)
        self.kb.clean()
        return self


class DrillHeuristic(AbstractScorer):
    """
    Heuristic in Convolutional DQL concept learning.
    Heuristic implements a convolutional neural network.
    """

    def __init__(self, pos=None, neg=None, model=None, mode=None, model_args=None):
        super().__init__(pos, neg, unlabelled=None)

        self.net = None
        self.model_args = None
        if model:
            self.net = model
        elif mode in ['averaging', 'sampling']:
            self.net = Drill(model_args)
            self.mode = mode
            self.name = 'DrillHeuristic_' + self.mode
            self.model_args = model_args
        elif mode in ['probabilistic']:
            self.net = DrillProba(model_args)
            self.mode = mode
            self.name = 'DrillHeuristic_' + self.mode
            self.model_args = model_args
        else:
            raise ValueError
        self.net.eval()

    def score(self, node, parent_node=None):
        """ Compute heuristic value of root node only"""
        if parent_node is None and node.is_root:
            return torch.FloatTensor([.0001]).squeeze()
        raise ValueError

    def apply(self, node, parent_node=None):
        """ Assign predicted Q-value to node object."""
        predicted_q_val = self.score(node, parent_node)
        node.heuristic = predicted_q_val


class Drill(nn.Module):
    """
    A neural model for Deep Q-Learning.

    An input Drill has the following form
            1. indexes of individuals belonging to current state (s).
            2. indexes of individuals belonging to next state state (s_prime).
            3. indexes of individuals provided as positive examples.
            4. indexes of individuals provided as negative examples.

    Given such input, we from a sparse 3D Tensor where  each slice is a **** N *** by ***D***
    where N is the number of individuals and D is the number of dimension of embeddings.
    Given that N on the current benchmark datasets < 10^3, we can get away with this computation. By doing so
    we do not need to subsample from given inputs.

    """

    def __init__(self, args):
        super(Drill, self).__init__()
        self.in_channels, self.embedding_dim = args['input_shape']
        self.loss = nn.MSELoss()

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=args['first_out_channels'],
                               kernel_size=args['kernel_size'],
                               padding=1, stride=1, bias=True)

        # Fully connected layers.
        self.size_of_fc1 = int(args['first_out_channels'] * self.embedding_dim)
        self.fc1 = nn.Linear(in_features=self.size_of_fc1, out_features=self.size_of_fc1 // 2)
        self.fc2 = nn.Linear(in_features=self.size_of_fc1 // 2, out_features=args['num_output'])

        self.init()
        assert self.__sanity_checking(torch.rand(32, 4, 1, self.embedding_dim)).shape == (32, 1)

    def init(self):
        xavier_normal_(self.fc1.weight.data)
        xavier_normal_(self.conv1.weight.data)

    def __sanity_checking(self, X):
        return self.forward(X)

    def forward(self, X: torch.FloatTensor):
        X = F.relu(self.conv1(X))
        X = X.view(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
        X = F.relu(self.fc1(X))
        return self.fc2(X)


class DrillProba(nn.Module):
    """
    A neural model for Deep Q-Learning.

    An input Drill has the following form
            1. indexes of individuals belonging to current state (s).
            2. indexes of individuals belonging to next state state (s_prime).
            3. indexes of individuals provided as positive examples.
            4. indexes of individuals provided as negative examples.

    Given such input, we form a sparse 3D Tensor where  each slice is a **** N *** by ***D***
    where N is the number of individuals and D is the number of dimension of embeddings.

    Outout => [0,1]
    """

    def __init__(self, args):
        super(DrillProba, self).__init__()
        self.in_channels, self.embedding_dim = args['input_shape']
        self.loss = nn.BCELoss()

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=args['first_out_channels'],
                               kernel_size=args['kernel_size'],
                               padding=1, stride=1, bias=True)

        # Fully connected layers.
        self.size_of_fc1 = int(args['first_out_channels'] * self.embedding_dim)
        self.fc1 = nn.Linear(in_features=self.size_of_fc1, out_features=self.size_of_fc1 // 2)
        self.fc2 = nn.Linear(in_features=self.size_of_fc1 // 2, out_features=args['num_output'])

        self.init()
        assert self.__sanity_checking(torch.rand(32, 4, 1, self.embedding_dim)).shape == (32, 1)

    def init(self):
        xavier_normal_(self.fc1.weight.data)
        xavier_normal_(self.conv1.weight.data)

    def __sanity_checking(self, X):
        return self.forward(X)

    def forward(self, X: torch.FloatTensor):
        X = F.relu(self.conv1(X))
        X = X.view(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
        X = F.relu(self.fc1(X))
        return torch.sigmoid(self.fc2(X))

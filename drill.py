from abc import ABCMeta
from collections import deque, OrderedDict
import owlapy.model
from abstracts import AbstractScorer

import time
import json
import pandas as pd
import numpy as np
import functools
from typing import Set, Tuple, List, Iterable
import torch
from torch import nn
from torch.functional import F
from torch.nn.init import xavier_normal_
import random
from reasoner import Nothing
from heuristics import BinaryReward, Reward
from metrics import F1


class State:
    def __init__(self, concept, previous_state=None):
        self.concept = concept
        self.previous_state = previous_state
        self.individuals = None

    def __str__(self):
        return f"<rl.State object at {hex(id(self))} containing {self.concept}>"

    def __len__(self):
        if isinstance(self.concept, owlapy.model.OWLClass):
            return 1
        elif isinstance(self.concept, owlapy.model.OWLObjectUnionOf) or isinstance(self.concept,
                                                                                   owlapy.model.OWLObjectIntersectionOf):
            length = 0
            for op in self.concept.operands():
                length += len(op)
            return length
        else:
            raise NotImplementedError('Type of the concept is not recognized')

    def __lt__(self, other):
        return len(self) < len(other)


from queue import PriorityQueue
from abc import abstractmethod


class DRILLAbstractTree:
    @abstractmethod
    def __init__(self):
        self._nodes = dict()

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, item):
        return self._nodes[item]

    def __setitem__(self, k, v):
        self._nodes[k] = v

    def __iter__(self):
        for k, node in self._nodes.items():
            yield node

    def get_top_n_nodes(self, n: int, key='quality'):
        self.sort_search_tree_by_decreasing_order(key=key)
        for ith, dict_ in enumerate(self._nodes.items()):
            if ith >= n:
                break
            k, node = dict_
            yield node

    def redundancy_check(self, n):
        if n in self._nodes:
            return False
        return True

    @property
    def nodes(self):
        return self._nodes

    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    def sort_search_tree_by_decreasing_order(self, *, key: str):
        if key == 'heuristic':
            sorted_x = sorted(self._nodes.items(), key=lambda kv: kv[1].heuristic, reverse=True)
        elif key == 'quality':
            sorted_x = sorted(self._nodes.items(), key=lambda kv: kv[1].quality, reverse=True)
        elif key == 'length':
            sorted_x = sorted(self._nodes.items(), key=lambda kv: len(kv[1]), reverse=True)
        else:
            raise ValueError('Wrong Key. Key must be heuristic, quality or concept_length')

        self._nodes = OrderedDict(sorted_x)

    def best_hypotheses(self, n=10) -> List:
        assert self.search_tree is not None
        assert len(self.search_tree) > 1
        return [i for i in self.search_tree.get_top_n_nodes(n)]

    def show_search_tree(self, th=0, top_n=10):
        """
        Show search tree.
        """
        print(f'######## {th}.step\t Top 10 nodes in Search Tree \t |Search Tree|={self.__len__()} ###########')
        predictions = list(self.get_top_n_nodes(top_n))
        for ith, node in enumerate(predictions):
            print(f'{ith + 1}-\t{node}')
        print('######## Search Tree ###########\n')
        return predictions

    def show_best_nodes(self, top_n, key=None):
        assert key
        self.sort_search_tree_by_decreasing_order(key=key)
        return self.show_search_tree('Final', top_n=top_n + 1)

    @staticmethod
    def save_current_top_n_nodes(key=None, n=10, path=None):

        """
        Save current top_n nodes
        """
        assert path
        assert key
        assert isinstance(n, int)
        pass

    def clean(self):
        self._nodes.clear()


class SearchTree(DRILLAbstractTree):
    """

    Search tree based on priority queue.

    Parameters
    ----------
    quality_func : An instance of a subclass of AbstractScorer that measures the quality of a node.
    heuristic_func : An instance of a subclass of AbstractScorer that measures the promise of a node.

    Attributes
    ----------
    quality_func : An instance of a subclass of AbstractScorer that measures the quality of a node.
    heuristic_func : An instance of a subclass of AbstractScorer that measures the promise of a node.
    items_in_queue: An instance of PriorityQueue Class.
    .nodes: A dictionary where keys are string representation of nodes and values are corresponding node objects.
    nodes: A property method for ._nodes.
    expressionTests: not being used .
    str_to_obj_instance_mapping: not being used.
    """

    def __init__(self):
        super().__init__()
        self.items_in_queue = PriorityQueue()

    def __len__(self):
        return len(self.items_in_queue.queue)

    def add(self, state: State, priority: float):
        """
        Append a node into the search tree.
        Parameters
        ----------
        state :
        priority:
        Returns
        -------
        None
        """
        self.items_in_queue.put((-priority, state))  # gets the smallest one.

    def get(self) -> Tuple:
        """
        Gets the current most promising node from Queue.

        Returns
        -------
        node: A node object
        """
        priority, state = self.items_in_queue.get(block=True, timeout=1.0)  # get
        return -priority, state

    def empty(self):
        return self.items_in_queue.empty()

    def get_top_n(self, n: int, key='quality') -> List[State]:
        """
        Gets the top n nodes determined by key from the search tree.

        Returns
        -------
        top_n_predictions: A list of node objects
        """
        all_nodes = self.refined_nodes + self.nodes.values()
        all_nodes.union(self.nodes)

        if key == 'quality':
            top_n_predictions = sorted(all_nodes, key=lambda node: node.quality, reverse=True)[:n]
        elif key == 'heuristic':
            top_n_predictions = sorted(all_nodes, key=lambda node: node.heuristic, reverse=True)[:n]
        elif key == 'length':
            top_n_predictions = sorted(self.nodes.values(), key=lambda node: len(node), reverse=True)[:n]
        else:
            print('Wrong Key:{0}\tProgram exist.'.format(key))
            raise KeyError
        return top_n_predictions

    def clean(self):
        self.items_in_queue = PriorityQueue()


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

        self.reasoner = reasoner
        self.epsilon = 0.1
        self.reward_func = Reward()
        self.num_episode = 100
        self.sample_size = 1
        self.embedding_dim = 32
        self.num_of_sequential_actions = 2
        self.drill_first_out_channels = 3
        self.learning_rate = .1
        self.max_len_replay_memory = 1024
        self.batch_size = 32
        self.num_workers: int = 0
        self.num_epochs_per_replay: int = 5
        self.epsilon: float = 1.0
        self.epsilon_decay: float = 0.001
        self.epsilon_min: float = .001
        self.relearn_ratio: int = 2
        self.emb_pos: torch.FloatTensor = None
        self.emb_neg: torch.FloatTensor = None
        self.experiences = Experience(maxlen=self.max_len_replay_memory)
        self.quality_func = quality_func
        self.search_tree = SearchTree()
        self.quality_tree = SearchTree()

        self.iter_bound = 2
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

    def next_possible_states(self, current_state: State) -> Iterable[State]:
        """ given a State, return all possible next states"""
        return (State(concept=i, previous_state=current_state) for i in
                self.reasoner.apply_construction_rules(current_state.concept))

    def exploitation(self, current_state: State, next_states: Set[State]) -> State:
        pass

    def choose_next_state(self, current_state: State, next_states: Set[State]) -> State:
        """
        Exploration vs Exploitation tradeoff at finding next state.
        (1) Exploration
        (2) Exploitation
        """
        # Sanity checking
        if len(next_states) == 0:  # DEAD END
            raise ValueError
        if np.random.random() < self.epsilon:
            next_state = random.choice(next_states)
        else:
            next_state = random.choice(next_states)
            # next_state = self.exploitation(current_state, next_states)
        return next_state

    def sequence_of_actions(self, root: State):
        current_state = root
        path_of_concepts = []
        rewards = []
        # (1)
        for _ in range(self.num_of_sequential_actions):
            # (1.1) Obtain possible next states
            next_states = self.next_possible_states(current_state)
            # (1.2) Select a next state
            next_state = self.choose_next_state(current_state, next_states)
            # (1.3)
            if isinstance(next_state.concept, Nothing):  # Dead END
                # (1.4)
                # path_of_concepts.append((current_state, next_state))
                # (1.5)
                # rewards.append(self.reward_func.calculate(current_state, next_state))
                print('Found dead end')
                break

            else:
                # (1.4)
                path_of_concepts.append((current_state, next_state))
                # (1.5)
                current_state.individuals = self.reasoner.retrieve(concept=current_state.concept)
                next_state.individuals = self.reasoner.retrieve(concept=next_state.concept)
                rewards.append(self.reward_func.calculate(current_state, next_state))
            # (1.6)
            current_state = next_state
        return path_of_concepts, rewards

    def form_experiences(self, state_pairs: List[State], rewards: List[float]) -> None:
        for th, consecutive_states in enumerate(state_pairs):
            e, e_next = consecutive_states
            # given e, e_next, Q val is the max Q value reachable.
            self.experiences.append(self.get_embeddings(e.individuals), self.get_embeddings(e_next.individuals),
                                    max(rewards[th:]))

    def learn_from_replay_memory(self) -> None:
        """
        Learning by replaying memory
        @return:
        """
        current_state_batch, next_state_batch, q_values = self.experiences.retrieve()
        current_state_batch = torch.cat(current_state_batch, dim=0)
        next_state_batch = torch.cat(next_state_batch, dim=0)
        q_values = torch.Tensor(q_values)
        dataset = PrepareBatchOfTraining(current_state_batch=current_state_batch,
                                         next_state_batch=next_state_batch,
                                         p=self.emb_pos, n=self.emb_neg, q=q_values)
        num_experience = len(dataset)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.batch_size, shuffle=True,
                                                  num_workers=self.num_workers)
        print(f'Number of experiences:{num_experience}\tDQL agent is learning via experience replay')
        self.heuristic_func.net.train()
        for m in range(self.num_epochs_per_replay):
            total_loss = 0
            for X, y in data_loader:
                self.optimizer.zero_grad()  # zero the gradient buffers
                # forward
                predicted_q = self.heuristic_func.net.forward(X)
                # loss
                loss = self.heuristic_func.net.loss(predicted_q, y)
                total_loss += loss.item()
                # compute the derivative of the loss w.r.t. the parameters using backpropagation
                loss.backward()
                # clip gradients if gradients are killed. =>torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
            print(f'{m}.th Epoch average loss during training:{total_loss / num_experience}')

        self.heuristic_func.net.train().eval()

    def get_embeddings_individuals(self, individuals: Iterable[str]):
        assert isinstance(individuals, set) and len(individuals) > 0
        return torch.rand(5).view(1, 5)

    def rl_learning_loop(self, pos, neg):
        # 2. Obtain embeddings of positive and negative examples.
        sum_of_rewards_per_actions = []
        root = State(concept=self.reasoner.thing)
        print('RL agent starts to interact with the environment. Trajectories will be summarized.')
        self.reward_func.pos = pos
        self.reward_func.neg = neg
        self.emb_pos = self.get_embeddings_individuals(pos)
        self.emb_neg = self.get_embeddings_individuals(neg)

        for th in range(1, self.num_episode + 1):
            sequence_of_states, rewards = self.sequence_of_actions(root)
            # (3.2) Form experiences for Experience Replay
            self.form_experiences(sequence_of_states, rewards)
            # (3.3) Experience Replay
            self.learn_from_replay_memory()
            # (3.4) Epsilon greedy => Exploration Exploitation
            self.epsilon -= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                break
        return sum_of_rewards_per_actions

    def train(self, learning_problems: Iterable[Tuple[Set[str], Set[str], Set[str], str]]) -> None:
        """
        An iterable of learning problems with an additional data
        1. A set of positive example individuals
        2. A set of negative example individuals
        3. A set of individuals corresponding to the answer set
        4. Target/Goal OWL class expression
        """
        for pos, neg, true_pos, owl_cls in learning_problems:
            for _ in range(self.relearn_ratio):
                sum_of_rewards_per_actions = self.rl_learning_loop(pos=pos, neg=neg)
                print(sum_of_rewards_per_actions)

    def next_states(self, state: State):
        # (1) Apply DL construction rules.
        for next_state in self.next_possible_states(state):
            # (2) Retrieve individuals.
            next_state.individuals = self.reasoner.retrieve(concept=next_state.concept)
            # No individuals
            if len(next_state.individuals) == 0:
                continue
            # (3) Retrieve embeddings of individuals.
            next_state.embeddings = self.get_embeddings_individuals(next_state.individuals)
            yield next_state

    def fit(self, pos: Set[str], neg: Set[str], ignore: Set[str] = None):
        """ Find hypotheses that explain pos and neg. """
        assert len(pos) > 1 and len(neg) > 1
        self.search_tree.clean()
        self.quality_tree.clean()
        # (1) Initialize the root state of the RL environment
        root = State(concept=self.reasoner.thing)
        root.individuals = self.reasoner.retrieve(concept=root.concept)
        root.embeddings = self.get_embeddings_individuals(root.individuals)

        # (2) Get embeddings for positive and negative examples.
        emb_pos = self.get_embeddings_individuals(pos)
        emb_neg = self.get_embeddings_individuals(neg)

        current_state = None
        # (3) Search starts
        print('Learning Problem')
        print("Pos:", pos)
        print("Neg:", neg)
        for i in range(self.iter_bound):
            if current_state is None:
                current_state = root
            else:
                # (1) Remove the best from the queue.
                _, current_state = self.search_tree.get()
            # Exploit
            for next_state in self.next_states(state=current_state):
                heuristic_score = self.heuristic_func.forward(e_state=current_state.embeddings,
                                                              e_next_state=next_state.embeddings,
                                                              e_pos=emb_pos,
                                                              e_neg=emb_neg)
                f1_score = self.quality_func(pos=pos, neg=neg, individuals=next_state.individuals)

                self.search_tree.add(state=next_state, priority=heuristic_score)
                self.quality_tree.add(state=next_state, priority=f1_score)
        return self.quality_tree.get()

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

    def score(self, *args, **kwargs):
        raise NotImplementedError()

    def apply(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, e_state: torch.FloatTensor, e_next_state: torch.FloatTensor, e_pos: torch.FloatTensor,
                e_neg: torch.FloatTensor) -> float:

        e_state = e_state.unsqueeze(1)
        e_next_state = e_next_state.unsqueeze(1)
        e_pos = e_pos.unsqueeze(1)
        e_neg = e_neg.unsqueeze(1)
        # n x input_channel x height x width
        # 1 x 4 x 1 x d
        X = torch.cat((e_state, e_next_state, e_pos, e_neg), 1).unsqueeze(2)
        return self.net.forward(X).item()


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
        self.loss = nn.MSELoss()
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=3,
                               kernel_size=3,
                               padding=1, stride=1, bias=False)
        # Fully connected layers.
        self.size_of_fc1 = int(args['first_out_channels'] * 5)
        self.fc1 = nn.Linear(in_features=self.size_of_fc1, out_features=self.size_of_fc1 // 2)
        self.fc2 = nn.Linear(in_features=self.size_of_fc1 // 2, out_features=args['num_output'])
        self.init()

    def init(self):
        xavier_normal_(self.fc1.weight.data)
        xavier_normal_(self.fc2.weight.data)
        xavier_normal_(self.conv1.weight.data)

    def forward(self, X: torch.FloatTensor):
        X = F.relu(self.conv1(X)).flatten(start_dim=1)
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


class Experience:
    """
    A class to model experiences for Replay Memory.
    """

    def __init__(self, maxlen: int):
        # @TODO we may want to not forget experiences yielding high rewards
        self.current_states_embeddings = deque(maxlen=maxlen)
        self.next_states_embeddings = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)

    def __len__(self):
        assert len(self.current_states) == len(self.next_states) == len(self.rewards)
        return len(self.current_states)

    def append(self, current_state: torch.FloatTensor, next_state: torch.FloatTensor, reward: float):
        """
        Embeddings of current state
        Embeddings of next steate
        """
        assert current_state.shape == next_state.shape
        self.current_states_embeddings.append(current_state)
        self.next_states_embeddings.append(next_state)
        self.rewards.append(reward)

    def retrieve(self):
        return list(self.current_states_embeddings), list(self.next_states_embeddings), list(self.rewards)

    def clear(self):
        self.current_states.clear()
        self.next_states.clear()
        self.rewards.clear()


class PrepareBatchOfTraining(torch.utils.data.Dataset):

    def __init__(self, current_state_batch: torch.FloatTensor, next_state_batch: torch.FloatTensor,
                 p: torch.FloatTensor,
                 n: torch.FloatTensor, q: torch.FloatTensor):
        """
        current_state_batch n by d matrix
        next_state_batch n by d matrix
        p k by d matrix
        n m by d matrix
        q row of n
        """
        # Sanity checking
        if torch.isnan(current_state_batch).any() or torch.isinf(current_state_batch).any():
            raise ValueError('invalid value detected in current_state_batch,\n{0}'.format(current_state_batch))
        if torch.isnan(next_state_batch).any() or torch.isinf(next_state_batch).any():
            raise ValueError('invalid value detected in next_state_batch,\n{0}'.format(next_state_batch))
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise ValueError('invalid value detected in p,\n{0}'.format(p))
        if torch.isnan(n).any() or torch.isinf(n).any():
            raise ValueError('invalid value detected in p,\n{0}'.format(n))
        if torch.isnan(q).any() or torch.isinf(q).any():
            raise ValueError('invalid Q value  detected during batching.')

        self.S = current_state_batch.unsqueeze(1)
        self.S_Prime = next_state_batch.unsqueeze(1)
        self.y = q.view(len(q), 1)
        assert self.S.shape == self.S_Prime.shape
        assert len(self.y) == len(self.S)
        try:
            self.Positives = p.expand(next_state_batch.shape).unsqueeze(1)
        except RuntimeError as e:
            print(p.shape)
            print(next_state_batch.shape)
            print(e)
            raise
        self.Negatives = n.expand(next_state_batch.shape).unsqueeze(1)

        self.X = torch.cat([self.S, self.S_Prime, self.Positives, self.Negatives], 1).unsqueeze(2)
        if torch.isnan(self.X).any() or torch.isinf(self.X).any():
            print('invalid input detected during batching in X')
            raise ValueError
        if torch.isnan(self.y).any() or torch.isinf(self.y).any():
            print('invalid Q value  detected during batching in Y')
            raise ValueError

        # number of data points x inputchannel x height x width
        # print(self.X.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

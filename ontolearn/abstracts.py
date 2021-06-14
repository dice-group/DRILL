from collections import OrderedDict, defaultdict
from functools import total_ordering
from abc import ABCMeta, abstractmethod, ABC
from owlready2 import Ontology
from .util import read_csv, performance_debugger
from typing import Set, Dict, List, Tuple, Iterable, Generator, SupportsFloat
import random
import pandas as pd
import torch
from .data_struct import PrepareBatchOfTraining, PrepareBatchOfPrediction, Experience
import json
import numpy as np
import time
import asyncio
from .static_funcs import retrieve_concept_chain
import torch.multiprocessing

random.seed(0)
# RuntimeError: Too many open files.
# Communication with the workers is no longer possible.
# Please increase the limit using `ulimit -n` in the shell or
# change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code.
torch.multiprocessing.set_sharing_strategy('file_system')


class BaseConcept(metaclass=ABCMeta):
    """Base class for Concept."""
    __slots__ = ['name', 'iri', 'form', 'is_atomic', 'length', '__instances', '__idx_instances', 'role', 'filler',
                 'concept_a', 'concept_b']

    @abstractmethod
    def __init__(self, name, iri, form, instances, concept_a, concept_b, role, filler):
        self.name = name
        self.iri = iri
        self.form = form
        self.is_atomic = True if self.form == 'Class' else False
        self.length = self.__calculate_length()
        self.__idx_instances = None
        self.__instances = instances

        # For concept concepts.
        self.role = role
        self.filler = filler
        self.concept_a = concept_a
        self.concept_b = concept_b
        self.__parse()

    def __parse(self):
        """
        :param kwargs:
        :return:
        """
        if self.iri is None:
            self.iri = self.name

        if not self.is_atomic:
            if self.form in ['ObjectSomeValuesFrom', 'ObjectAllValuesFrom']:
                assert isinstance(self.filler, BaseConcept)
            elif self.form in ['ObjectUnionOf', 'ObjectIntersectionOf']:
                assert isinstance(self.concept_a, BaseConcept)
                assert isinstance(self.concept_b, BaseConcept)
            elif self.form in ['ObjectComplementOf']:
                assert isinstance(self.concept_a, BaseConcept)
            else:
                print('Wrong type')
                print(self)
                raise ValueError

    @property
    def instances(self) -> Set:
        """ Returns all instances belonging to the concept."""
        return self.__instances

    @property
    def num_instances(self):
        if self.__instances is not None:
            return len(self.__instances)
        return None

    @instances.setter
    def instances(self, x: Set):
        """ Setter of instances."""
        self.__instances = x

    @property
    def idx_instances(self):
        """ Getter of integer indexes of instances."""
        return self.__idx_instances

    @idx_instances.setter
    def idx_instances(self, x):
        """ Setter of idx_instances."""
        self.__idx_instances = x

    def __str__(self):
        return '{self.iri}'.format(self=self)

    def __repr__(self):
        return '{self.iri}'.format(self=self)

    def __len__(self):
        return self.length

    def __calculate_length(self):
        """
        The length of a concept is defined as
        the sum of the numbers of
            concept names, role names, quantifiers,and connective symbols occurring in the concept

        The length |A| of a concept CAis defined inductively:
        |\top| = |\bot| = 1
        |¬D| = |D| + 1
        |D \sqcap E| = |D \sqcup E| = 1 + |D| + |E|
        |∃r.D| = |∀r.D| = 2 + |D|
        :return:
        """
        num_of_exists = self.name.count("∃")
        num_of_for_all = self.name.count("∀")
        num_of_negation = self.name.count("¬")
        is_dot_here = self.name.count('.')

        num_of_operand_and_operator = len(self.name.split())
        count = num_of_negation + num_of_operand_and_operator + num_of_exists + is_dot_here + num_of_for_all
        return count

    def __is_atomic(self):
        if '∃' in self.name or '∀' in self.name:
            return False
        elif '⊔' in self.name or '⊓' in self.name or '¬' in self.name:
            return False
        return True

    def __lt__(self, other):
        return self.length < other.length

    def __gt__(self, other):
        return self.length > other.length


@total_ordering
class BaseNode(metaclass=ABCMeta):
    """Base class for Concept."""
    __slots__ = ['concept', '__heuristic_score', '__horizontal_expansion', '__quality_score',
                 '___refinement_count', '__refinement_count', '__depth', '__children', '__embeddings', 'length',
                 'parent_node']

    @abstractmethod
    def __init__(self, concept, parent_node, is_root=False):
        self.__quality_score, self.__heuristic_score = None, None
        self.__is_root = is_root
        self.__horizontal_expansion, self.__refinement_count = 0, 0
        self.concept = concept
        self.parent_node = parent_node
        self.__embeddings = None
        self.__children = set()
        self.length = len(self.concept)

        if self.parent_node is None:
            assert len(concept) == 1 and self.__is_root
            self.__depth = 0
        else:
            self.__depth = self.parent_node.depth + 1

    def __len__(self):
        return len(self.concept)

    @property
    def embeddings(self):
        return self.__embeddings

    @embeddings.setter
    def embeddings(self, value):
        self.__embeddings = value

    @property
    def children(self):
        return self.__children

    @property
    def refinement_count(self):
        return self.__refinement_count

    @refinement_count.setter
    def refinement_count(self, n):
        self.__refinement_count = n

    @property
    def depth(self):
        return self.__depth

    @depth.setter
    def depth(self, n: int):
        self.__depth = n

    @property
    def h_exp(self):
        return self.__horizontal_expansion

    @property
    def heuristic(self) -> float:
        return self.__heuristic_score

    @heuristic.setter
    def heuristic(self, val: float):
        self.__heuristic_score = val

    @property
    def quality(self) -> float:
        return self.__quality_score

    @quality.setter
    def quality(self, val: float):
        self.__quality_score = val

    @property
    def is_root(self):
        return self.__is_root

    def add_children(self, n):
        self.__children.add(n)

    def remove_child(self, n):
        self.__children.remove(n)

    def increment_h_exp(self, val=0):
        self.__horizontal_expansion += val + 1

    def __lt__(self, other):
        return self.concept.length < other.concept.length

    def __gt__(self, other):
        return self.concept.length > other.concept.length


class AbstractScorer(ABC):
    @abstractmethod
    def __init__(self, pos, neg, unlabelled):
        self.pos = pos
        self.neg = neg
        self.unlabelled = unlabelled
        self.applied = 0

    def set_positive_examples(self, instances):
        self.pos = instances

    def set_negative_examples(self, instances):
        self.neg = instances

    def set_unlabelled_examples(self, instances):
        self.unlabelled = instances

    @abstractmethod
    def score(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

    def clean(self):
        self.pos = None
        self.neg = None
        self.unlabelled = None
        self.applied = 0

    @property
    def num_times_applied(self):
        return self.applied


class BaseRefinement(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, kb, max_size_of_concept=10_000, min_size_of_concept=0):
        self.kb = kb
        self.max_size_of_concept = max_size_of_concept
        self.min_size_of_concept = min_size_of_concept

    def set_kb(self, kb):
        self.kb = kb

    @abstractmethod
    def getNode(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_atomic_concept(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_complement_of(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_object_some_values_from(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_object_all_values_from(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_object_union_of(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_object_intersection_of(self, *args, **kwargs):
        pass


class AbstractTree(ABC):
    @abstractmethod
    def __init__(self, quality_func, heuristic_func):
        self.quality_func = quality_func
        self.heuristic_func = heuristic_func
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

    def set_quality_func(self, f: AbstractScorer):
        self.quality_func = f

    def set_heuristic_func(self, h):
        self.heuristic_func = h

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

    def best_hypotheses(self, n=10) -> List[BaseNode]:
        assert self.search_tree is not None
        assert len(self.search_tree) > 1
        return [i for i in self.search_tree.get_top_n_nodes(n)]

    def show_search_tree(self, th, top_n=10):
        """
        Show search tree.
        """
        print(f'######## {th}.step\t Top 10 nodes in Search Tree \t |Search Tree|={self.__len__()} ###########')
        predictions = list(self.get_top_n_nodes(top_n))
        for ith, node in enumerate(predictions):
            print('{0}-\t{1}\t{2}:{3}\tHeuristic:{4}:'.format(ith + 1, node.concept.name,
                                                              self.quality_func.name, node.quality, node.heuristic))
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


class AbstractKnowledgeBase(ABC):

    def __init__(self):
        self.uri_to_concept = dict()
        self.thing = None
        self.nothing = None
        self.top_down_concept_hierarchy = defaultdict(set)  # Next time thing about including this into Concepts.
        self.top_down_direct_concept_hierarchy = defaultdict(set)
        self.down_top_concept_hierarchy = defaultdict(set)
        self.down_top_direct_concept_hierarchy = defaultdict(set)
        self.concepts_to_leafs = defaultdict(set)
        self.property_hierarchy = None
        self.individuals = None
        self.uri_individuals = None  # string representation of uris

    def save(self, path: str, rdf_format="nt"):
        """
        @param path: xxxx.nt
        @param rdf_format:
        @return:
        """
        # self.onto.save(file=path, format=rdf_format) => due to world object it only creates empty file.
        self.world.as_rdflib_graph().serialize(destination=path + '.' + rdf_format, format=rdf_format)

    def describe(self):
        print('#' * 10)
        print(f'Knowledge Base: {self.name}\n'
              f'Number of concepts: {len(self.uri_to_concept)}\n'
              f'Number of individuals: {len(self.individuals)}\n'
              f'Number of properties: {len(self.property_hierarchy.all_properties)}\n'
              f'Number of data properties: {len(self.property_hierarchy.data_properties)}\n'
              f'Number of object properties: {len(self.property_hierarchy.object_properties)}')
        print('#' * 10)

    def __str__(self):
        return f'Knowledge Base:{self.name}'

    @abstractmethod
    def clean(self):
        raise NotImplementedError

    @property
    def concepts(self):
        return [i for i in self.uri_to_concept.values()]


class AbstractDrill(ABC):
    """
    Abstract class for Convolutional DQL concept learning
    """

    def __init__(self, path_of_embeddings, reward_func, drill_first_out_channels=32, gamma=1.0, learning_rate=.001,
                 num_episode=None, num_of_sequential_actions=2, max_len_replay_memory=1024,
                 relearn_ratio=1, representation_mode=None, batch_size=1024, epsilon_decay=.001,
                 num_epochs_per_replay=50, num_episodes_per_replay=25, num_workers=32):
        self.drill_first_out_channels = drill_first_out_channels
        self.instance_embeddings = read_csv(path_of_embeddings)
        self.embedding_dim = self.instance_embeddings.shape[1]
        self.num_episodes_per_replay = num_episodes_per_replay
        self.reward_func = reward_func
        self.gamma = gamma
        assert reward_func
        self.representation_mode = representation_mode
        assert representation_mode in ['averaging', 'sampling']
        # Will be filled by child class
        self.heuristic_func = None
        self.num_workers = num_workers
        # constants
        self.epsilon = 1
        if learning_rate is None:
            self.learning_rate = .01
        else:
            self.learning_rate = learning_rate
        self.num_episode = num_episode
        self.num_of_sequential_actions = num_of_sequential_actions
        self.num_epochs_per_replay = num_epochs_per_replay
        self.max_len_replay_memory = max_len_replay_memory
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0
        self.batch_size = batch_size
        self.relearn_ratio = relearn_ratio

        # will be filled
        self.optimizer = None  # e.g. torch.optim.Adam(self.model_net.parameters(), lr=self.learning_rate)

        self.seen_examples = dict()
        self.emb_pos, self.emb_neg = None, None
        self.start_time = None
        self.goal_found = False
        self.experiences = Experience(maxlen=self.max_len_replay_memory)

    def default_state_rl(self):
        self.emb_pos, self.emb_neg = None, None
        self.goal_found = False
        self.start_time = None

    @abstractmethod
    def init_training(self, *args, **kwargs):
        """
        Initialize training for a given E+,E- and K.
        @param args:
        @param kwargs:
        @return:
        """

    @abstractmethod
    def terminate_training(self):
        """
        Save weights and training data after training phase.
        @return:
        """

    def next_node_to_expand(self, t: int = None) -> BaseNode:
        """
        Return a node that maximizes the heuristic function at time t
        @param t:
        @return:
        """
        if self.verbose > 1:
            self.search_tree.show_search_tree(t)
        return self.search_tree.get_most_promising()

    def form_experiences(self, state_pairs: List, rewards: List) -> None:
        """
        Form experiences from a sequence of concepts and corresponding rewards.

        state_pairs - a list of tuples containing two consecutive states, [(Node,Node),...,((Node,Node))]
        reward      - a list of reward.

        """
        assert len(state_pairs) == len(rewards)

        for th, consecutive_states in enumerate(state_pairs):
            s_i, s_j = consecutive_states
            # 1. Immediate rewards, [r_0,r_1,r_2]
            current_and_future_rewards = rewards[th:]
            # 2. [\gamma^0,\gamma^1,\gamma^2]
            aligned_discount_rates = np.power(self.gamma, np.arange(len(current_and_future_rewards)))
            # 3. G_t = \gamma^0* r_t + \gamma^1 r_t+1 + + \gamma^2 r_t+2
            discounted_reward = sum(current_and_future_rewards * aligned_discount_rates)
            self.experiences.append((s_i, s_j, discounted_reward))

    def learn_from_replay_memory(self) -> None:
        """
        Learning by replaying memory
        @return:
        """

        current_state_batch, next_state_batch, q_values = self.experiences.retrieve()
        current_state_batch = torch.cat(current_state_batch, dim=0)
        next_state_batch = torch.cat(next_state_batch, dim=0)
        q_values = torch.Tensor(q_values)

        try:
            assert current_state_batch.shape[1] == next_state_batch.shape[1] == self.emb_pos.shape[1] == \
                   self.emb_neg.shape[1]

        except AssertionError as e:
            print(current_state_batch.shape)
            print(next_state_batch.shape)
            print(self.emb_pos.shape)
            print(self.emb_neg.shape)
            print('Wrong format.')
            print(e)
            exit(1)

        assert current_state_batch.shape[2] == next_state_batch.shape[2] == self.emb_pos.shape[2] == self.emb_neg.shape[
            2]
        dataset = PrepareBatchOfTraining(current_state_batch=current_state_batch,
                                         next_state_batch=next_state_batch,
                                         p=self.emb_pos, n=self.emb_neg, q=q_values)
        num_experience = len(dataset)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.batch_size, shuffle=True,
                                                  num_workers=self.num_workers)
        print(f'Optimizing weights of the agent on {num_experience} number of experiences')
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
        self.heuristic_func.net.train().eval()

    def sequence_of_actions(self, root: BaseNode) -> Tuple[List[Tuple[BaseNode, BaseNode]], List[SupportsFloat]]:
        """
        Perform self.num_of_sequential_actions number of actions

        (1) Make a sequence of **self.num_of_sequential_actions** actions
            (1.1) Get next states in a generator and convert them to list
            (1.2) Exit, if If there is no next state.
            (1.3) Find next state.
            (1.4) Exit, if next state is **Nothing**
            (1.5) Compute reward.
            (1.6) Update current state.

        (2) Return path_of_concepts, rewards

        """
        assert isinstance(root, BaseNode)

        current_state = root
        path_of_concepts = []
        rewards = []
        # (1)
        for _ in range(self.num_of_sequential_actions):
            # (1.1)
            next_states = list(self.apply_rho(current_state))
            # (1.2)
            next_state = self.exploration_exploitation_tradeoff(current_state, next_states)
            # (1.3)
            if next_state.concept.name == 'Nothing':  # Dead END
                # (1.4)
                path_of_concepts.append((current_state, next_state))
                # (1.5)
                rewards.append(self.reward_func.calculate(current_state, next_state))
                break
            # (1.4)
            path_of_concepts.append((current_state, next_state))
            # (1.5)
            rewards.append(self.reward_func.calculate(current_state, next_state))
            # (1.6)
            current_state = next_state
        # (2)
        return path_of_concepts, rewards

    def update_search(self, concepts, predicted_Q_values):
        """
        @param concepts:
        @param predicted_Q_values:
        @return:
        """
        # This can be done in parallel.
        for child_node, pred_Q in zip(concepts, predicted_Q_values):
            child_node.heuristic = pred_Q
            if child_node.quality > 0:
                self.search_tree.add(child_node)
            else:
                """Ignore node containing a class expression that has 0.0 quality """

            if child_node.quality == 1:
                return child_node

    def apply_rho(self, node: BaseNode) -> Generator:
        assert isinstance(node, BaseNode)
        for i in self.rho.refine(node):  # O(N)
            if i.name not in self.concepts_to_ignore:  # O(1)
                yield self.rho.getNode(i, parent_node=node)  # O(1)

    def save_weights(self):
        """
        Save pytorch weights.
        @return:
        """
        # Save model.
        torch.save(self.heuristic_func.net.state_dict(),
                   self.storage_path + '/{0}.pth'.format(self.heuristic_func.name))

    def learn_from_illustration(self, sequence_of_goal_path):
        current_state = sequence_of_goal_path.pop(0)
        rewards = []
        sequence_of_states = []
        while len(sequence_of_goal_path) > 0:
            self.assign_embeddings(current_state)

            next_state = sequence_of_goal_path.pop(0)
            self.assign_embeddings(next_state)
            sequence_of_states.append((current_state, next_state))

            rewards.append(self.reward_func.calculate(current_state, next_state))
            current_state = next_state

        for x in range(2):
            self.form_experiences(sequence_of_states, rewards)
        self.learn_from_replay_memory()

    def rl_learning_loop(self, pos_uri: Set[str], neg_uri: Set[str], goal_path: List[BaseNode] = None) -> \
            List[float]:
        """
        RL agent learning loop over learning problem defined
        @param pos_uri: A set of URIs indicating E^+
        @param neg_uri: A set of URIs indicating E^-
        @param goal_path: an iterable indicating a goal path (optinal).

        Computation


        @return: List of sum of rewards per episode.
        """

        # (1) Initialize training.
        self.init_training(pos_uri=pos_uri, neg_uri=neg_uri)
        root = self.rho.getNode(self.start_class, root=True)
        self.assign_embeddings(root)
        sum_of_rewards_per_actions = []
        log_every_n_episodes = int(self.num_episode / 10) + 1

        # (2) Goal trajectory demonstration
        if goal_path:
            self.learn_from_illustration(goal_path)

        # (3) Training starts.
        for th in range(1, self.num_episode + 1):
            # (3.1) Search => Take sequence of actions.
            sequence_of_states, rewards = self.sequence_of_actions(root)
            if th % log_every_n_episodes == 1:
                print(
                    '{0}.th episode. SumOfRewards: {1:.2f}\tEpsilon:{2:.2f}\t|ReplayMem.|:{3}'.format(th, sum(rewards),
                                                                                                      self.epsilon, len(
                            self.experiences)))
                self.logger.info(
                    '{0}.th episode. SumOfRewards: {1:.2f}\tEpsilon:{2:.2f}\t|ReplayMem.|:{3}'.format(th, sum(rewards),
                                                                                                      self.epsilon, len(
                            self.experiences)))

            # (3.2) Form experiences for Experience Replay
            self.form_experiences(sequence_of_states, rewards)
            sum_of_rewards_per_actions.append(sum(rewards))

            # (3.3) Experience Replay
            if th % self.num_episodes_per_replay == 0:
                self.learn_from_replay_memory()

            # (3.4) Epsilon greedy => Exploration Exploitation
            self.epsilon -= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                break

        return sum_of_rewards_per_actions

    def exploration_exploitation_tradeoff(self, current_state: BaseNode, next_states: List[BaseNode]) -> BaseNode:
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
            self.assign_embeddings(next_state)
        else:
            next_state = self.exploitation(current_state, next_states)

        assert isinstance(next_state.embeddings, torch.Tensor)

        return next_state

    def exploitation(self, current_state: BaseNode, next_states: List[BaseNode]) -> BaseNode:
        """
        Find next node that is assigned with highest predicted Q value.

        (1) Predict Q values : predictions.shape => torch.Size([n, 1]) where n = len(next_states)

        (2) Find the index of max value in predictions

        (3) Use the index to obtain next state.

        (4) Return next state.
        """
        predictions: torch.Tensor = self.predict_Q(current_state, next_states)
        argmax_id = int(torch.argmax(predictions))
        next_state = next_states[argmax_id]
        """
        # Sanity checking
        print('#'*10)
        for s, q in zip(next_states, predictions):
            print(s, q)
        print('#'*10)
        print(next_state,f'\t {torch.max(predictions)}')
        """
        self.assign_embeddings(next_state)
        return next_state

    def assign_embeddings(self, node: BaseNode) -> None:
        if node.concept.instances is None:
            node.concept.instances = self.kb.instance_retrieval(node.concept)

        # (1) Detect mode
        if self.representation_mode == 'averaging':
            str_idx = [i for i in node.concept.instances]
            if len(str_idx) == 0:
                emb = torch.zeros(self.sample_size, self.instance_embeddings.shape[1])
            else:
                emb = torch.tensor(self.instance_embeddings.loc[str_idx].values, dtype=torch.float32)
                emb = torch.mean(emb, dim=0)
            emb = emb.view(1, self.sample_size, self.instance_embeddings.shape[1])
            node.embeddings = emb
        elif self.representation_mode == 'sampling':
            str_idx = [i for i in node.concept.instances]
            if len(str_idx) >= self.sample_size:
                sampled_str_idx = random.sample(str_idx, self.sample_size)
                emb = torch.tensor(self.instance_embeddings.loc[sampled_str_idx].values, dtype=torch.float32)
            else:
                num_rows_to_fill = self.sample_size - len(str_idx)
                emb = torch.tensor(self.instance_embeddings.loc[str_idx].values, dtype=torch.float32)
                emb = torch.cat((torch.zeros(num_rows_to_fill, self.instance_embeddings.shape[1]), emb))
            emb = emb.view(1, self.sample_size, self.instance_embeddings.shape[1])
            node.embeddings = emb

        else:
            raise ValueError

    @performance_debugger('get_instances_parallel')
    def get_instances_from_iterable(self, nodes: List[BaseNode]) -> List[List[str]]:
        return self.kb.instance_retrieval_parallel_from_iterable(nodes)

    @staticmethod
    def get_embeddings(state: BaseNode, instances: List[str], instance_embeddings: pd.DataFrame,
                       representation_mode: str):
        if representation_mode == 'averaging':
            if len(instances) == 0:
                # emb = torch.zeros(1, sample_size, instance_embeddings.shape[1])
                emb = np.zeros(instance_embeddings.shape[1])
            else:
                emb = instance_embeddings.loc[instances].values.mean(axis=0)
                # emb = torch.tensor(instance_embeddings.loc[instances].values, dtype=torch.float32)
                # emb = torch.mean(emb, dim=0)
                # emb = emb.view(1, sample_size, instance_embeddings.shape[1])
        else:
            # Later averaging.
            raise ValueError
        # state.embeddings = emb
        state.concept.instances = set(instances)
        return emb

    """
    In future, we plan to further decrease runtimes. To this end, we investigate asynchronous programming.
    @staticmethod
    async def loop(c):
        return await asyncio.gather(*c)

    @staticmethod
    async def async_get_embeddings(state: BaseNode, instances: List[str]):
        if representation_mode == 'averaging':
            if len(instances) == 0:
                emb = np.zeros(instance_embeddings.shape[1])
            else:
                emb = instance_embeddings.loc[instances].values.mean(axis=0)
        else:
            # Later averaging.
            raise ValueError
        state.embeddings = emb
        state.concept.instances = set(instances)
        return emb
    """

    @performance_debugger('assign_embeddings_in_parallel')
    def get_embeddings_from_iterable(self, batch_of_states: List[BaseNode], batch_instances: List[List[str]]):

        result = []
        for state, instances in zip(batch_of_states, batch_instances):
            result.append(self.get_embeddings(state, instances, self.instance_embeddings, self.representation_mode))

        result = torch.FloatTensor(result)
        return result.view(len(result), self.sample_size, self.instance_embeddings.shape[1])

    @performance_debugger('prepare_batch_in_parallel')
    def prepare_batch(self, batch_of_states: List[BaseNode]):
        """
        1. Get all instances.
        2. Get embeddings of instances.
        """
        batch_of_instances = self.get_instances_from_iterable(batch_of_states)
        next_state_batch = self.get_embeddings_from_iterable(batch_of_states, batch_of_instances)
        return next_state_batch

    @performance_debugger('predict_Q')
    def predict_Q(self, current_state: BaseNode, next_states: List[BaseNode]) -> torch.Tensor:
        """
        Predict promise of next states given current state.
        @param current_state:
        @param next_states:
        @return: predicted Q values.
        """
        try:
            assert len(next_states) > 0
        except AssertionError:
            print(f'DEAD END at {current_state}')
            raise
        self.assign_embeddings(current_state)
        with torch.no_grad():
            self.heuristic_func.net.eval()
            next_state_batch = self.prepare_batch(next_states)
            ds = PrepareBatchOfPrediction(current_state.embeddings,
                                          next_state_batch,
                                          self.emb_pos,
                                          self.emb_neg)
            predictions = self.heuristic_func.net.forward(ds.get_all())
        return predictions

    def train(self, dataset: Iterable[Tuple[str, Set, Set]]):
        self.logger.info('Training starts.')
        print(f'Training starts.\nNumber of learning problem:{len(dataset)},\t Relearn ratio:{self.relearn_ratio}')
        counter = 1
        # 1.
        for _ in range(self.relearn_ratio):
            for (target_node, positives, negatives) in dataset:
                print('\nGoal Concept:{0}\tE^+:[{1}] \t E^-:[{2}]'.format(target_node.concept.name,
                                                                          len(positives), len(negatives)))
                self.logger.info(
                    'Goal Concept:{0}\tE^+:[{1}] \t E^-:[{2}]'.format(target_node.concept.name,
                                                                      len(positives), len(negatives)))
                # 2.
                print(f'RL training on {counter}.th learning problem starts')
                goal_path = list(reversed(retrieve_concept_chain(target_node)))
                sum_of_rewards_per_actions = self.rl_learning_loop(pos_uri=positives, neg_uri=negatives,
                                                                   goal_path=goal_path)

                print(f'Sum of Rewards in first 3 trajectory:{sum_of_rewards_per_actions[:3]}')
                print(f'Sum of Rewards in last 3 trajectory:{sum_of_rewards_per_actions[-3:]}')
                self.seen_examples.setdefault(counter, dict()).update(
                    {'Concept': target_node.concept.name, 'Positives': list(positives), 'Negatives': list(negatives)})

                counter += 1
                if counter % 10 == 0:
                    self.save_weights()
                # 3.
        return self.terminate_training()

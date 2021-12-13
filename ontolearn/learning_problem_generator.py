import random
from typing import List, Any, Generator, Dict, Set
from .refinement_operators import LengthBasedRefinement
from queue import PriorityQueue
import itertools
import sys
from .util import balanced_sets, performance_debugger
from collections import deque
import numpy as np
from .static_funcs import retrieve_concept_chain


class LearningProblemGenerator:
    """ Learning problem generator. """

    def __init__(self, knowledge_base, refinement_operator=None, num_problems=10_000, num_diff_runs=100,
                 min_num_instances=None, max_num_instances=sys.maxsize, min_length=3, max_length=5, depth=10,
                 search_algo='strict-dfs'):
        """
        Generate concepts via search algorithm to satisfy constraints.
         strict-dfs considers (min_length, max_length, min_num_ind, num_problems) as hard constraints.
         dfs- considers (min_length, max_length, min_num_ind) as hard constraints and soft (>=num_problems).

         Trade-off between num_diff_runs and num_problems.
        """
        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(kb=knowledge_base)

        if max_num_instances and min_num_instances:
            assert max_num_instances >= min_num_instances

        if max_length and min_length:
            assert max_length >= min_length
        self.kb = knowledge_base
        self.rho = refinement_operator
        self.search_algo = search_algo
        self.min_num_instances = min_num_instances
        self.max_num_instances = max_num_instances
        self.min_length = min_length
        self.max_length = max_length
        self.valid_learning_problems = []
        self.depth = depth
        self.num_diff_runs = num_diff_runs
        self.num_problems = num_problems // self.num_diff_runs

    def get_balanced_n_samples_per_examples(self, *, n=5, min_num_problems=None, max_length=None, min_length=None,
                                            num_diff_runs=None, min_num_instances=None,
                                            search_algo='strict-dfs') -> list:
        """
        1. We generate min_num_problems number of concepts
        2. For each concept, we generate n number of positive and negative examples
        3. Each example contains
        @param n:
        @param min_num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_instances:
        @param search_algo:
        @return:
        """

        def concept_sanity_check(x):
            try:
                assert len(x.concept.instances)
            except AssertionError:
                print(f'{x}\nDoes not contain any instances. No instance to balance. Exiting.')
                raise

        assert self.min_num_instances or min_num_instances

        res = []
        gen_examples = []
        for example_node in self.generate_examples(num_problems=min_num_problems, max_length=max_length,
                                                   min_length=min_length, num_diff_runs=num_diff_runs,
                                                   min_num_instances=min_num_instances, search_algo=search_algo):
            concept_sanity_check(example_node)
            gen_examples.append(example_node)
            for d in self.balanced_n_sampled_lp(n, example_node.concept.instances):
                assert len(d['string_balanced_pos']) == len(d['string_balanced_neg'])
                res.append((example_node, d['string_balanced_pos'], d['string_balanced_neg']))
        try:
            assert len(gen_examples) > 0
        except AssertionError:
            print('*****No examples are created. Please update the configurations for learning problem generator****')
            exit(1)

        stats = np.array([[len(x), len(x.concept.instances)] for x in gen_examples])

        print(f'\nNumber of generated concepts:{len(gen_examples)}')
        print(f'Number of generated learning problems via sampling: {len(res)}')
        print(
            f'Average length of generated concepts:{stats[:, 0].mean():.3f}\nAverage number of individuals belong to a generated example:{stats[:, 1].mean():.3f}\n')
        return res

    def balanced_n_sampled_lp(self, n: int, string_all_pos: set):

        string_all_neg = self.kb.individuals.difference(string_all_pos)
        for i in range(n):
            string_balanced_pos, string_balanced_neg = balanced_sets(string_all_pos, string_all_neg)
            assert len(string_balanced_pos) >= self.min_num_instances
            assert len(string_balanced_neg) >= self.min_num_instances

            yield {'string_balanced_pos': string_balanced_pos, 'string_balanced_neg': string_balanced_neg}

    def get_balanced_examples(self, *, min_num_problems=None, max_length=None, min_length=None,
                              num_diff_runs=None, min_num_instances=None, search_algo='strict-dfs') -> list:
        """
        (1) Generate valid examples with input search algorithm.
        (2) Balance valid examples.

        @param min_num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_instances:
        @param search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as a hard constrain.
        @return: A list of tuples (s,p,n) where s denotes the string representation of a concept,
        p and n denote a set of URIs of individuals indicating positive and negative examples.

        """

        def output_sanity_check(y):
            try:
                assert len(y) >= min_num_problems
            except AssertionError:
                print('Not enough number of problems are generated')
                raise

        assert self.min_num_instances or min_num_instances

        res = []
        gen_examples = []
        for example_node in self.generate_examples(num_problems=min_num_problems, max_length=max_length,
                                                   min_length=min_length, num_diff_runs=num_diff_runs,
                                                   min_num_instances=min_num_instances, search_algo=search_algo):
            d = self.owlready_individuals_to_string_balanced_examples(example_node.concept.instances)
            res.append((example_node.concept.str, d['string_balanced_pos'], d['string_balanced_neg']))
            gen_examples.append(example_node)

        output_sanity_check(res)
        stats = np.array([[len(x), len(x.concept.instances)] for x in gen_examples])
        print(
            f'Average length of generated examples {stats[:, 0].mean():.3f}\nAverage number of individuals belong to a generated example {stats[:, 1].mean():.3f}')
        return res

    def get_examples(self, *, num_problems=None, max_length=None, min_length=None,
                     num_diff_runs=None, min_num_ind=None, search_algo=None) -> list:
        """
        (1) Get valid examples with input search algorithm.

        @param num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_ind:
        @param search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as hard constriant.
        @return: A list of tuples (s,p,n) where s denotes the string representation of a concept,
        p and n denote a set of URIs of individuals indicating positive and negative examples.

        """
        res = []
        for example_node in self.generate_examples(num_problems=num_problems,
                                                   max_length=max_length, min_length=min_length,
                                                   num_diff_runs=num_diff_runs, min_num_instances=min_num_ind,
                                                   search_algo=search_algo):
            try:
                assert len(example_node.concept.instances)
            except AssertionError as e:
                print(e)
                print(f'{example_node}\nDoes not contain any instances. No instance to balance. Exiting.')
                exit(1)
            string_all_pos = set(
                self.kb.convert_owlready2_individuals_to_uri_from_iterable(example_node.concept.instances))
            string_all_neg = set(self.kb.convert_owlready2_individuals_to_uri_from_iterable(
                self.kb.individuals.difference(example_node.concept.instances)))
            res.append((example_node.concept.str, string_all_pos, string_all_neg))
        return res

    def get_concepts(self, *, num_problems=None, max_length=None, min_length=None,
                     max_num_instances=None, num_diff_runs=None, min_num_instances=None, search_algo=None) -> Generator:
        """
        @param max_num_instances:
        @param num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_instances:
        @param search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as hard constriant.
        @return: A list of tuples (s,p,n) where s denotes the string representation of a concept,
        p and n denote a set of URIs of individuals indicating positive and negative examples.
        """
        return self.generate_examples(num_problems=num_problems,
                                      max_length=max_length, min_length=min_length,
                                      num_diff_runs=num_diff_runs,
                                      max_num_instances=max_num_instances,
                                      min_num_instances=min_num_instances,
                                      search_algo=search_algo)

    def generate_examples(self, *, num_problems=None, max_length=None, min_length=None,
                          num_diff_runs=None, max_num_instances=None, min_num_instances=None,
                          search_algo=None) -> Generator:
        """
        Generate examples via search algorithm that are valid examples w.r.t. given constraints

        @param num_diff_runs:
        @param num_problems:
        @param max_length:
        @param min_length:
        @param min_num_instances:
        @param max_num_instances:
        @param search_algo:
        @return:
        """
        if num_problems and num_diff_runs:
            assert isinstance(num_problems, int)
            assert isinstance(num_diff_runs, int)
            self.num_problems = num_problems
            self.num_diff_runs = num_diff_runs
            self.num_problems //= self.num_diff_runs
        elif num_diff_runs:
            assert isinstance(num_diff_runs, int)
            self.num_diff_runs = num_diff_runs
            self.num_problems //= self.num_diff_runs
        elif num_problems:
            assert isinstance(num_problems, int)
            self.num_problems = num_problems // self.num_diff_runs

        if max_length:
            assert isinstance(max_length, int)
            self.max_length = max_length
        if min_length:
            assert isinstance(min_length, int)
            self.min_length = min_length
        if min_num_instances:
            assert isinstance(min_num_instances, int)
            self.min_num_instances = min_num_instances
            # Do not generate concepts that do not define enough individuals.
            self.rho.min_num_instances = self.min_num_instances

        if max_num_instances:
            assert isinstance(max_num_instances, int)
            self.max_num_instances = max_num_instances
        if search_algo:
            self.search_algo = search_algo

        if self.min_num_instances and self.max_num_instances == sys.maxsize:
            assert isinstance(self.min_num_instances, int)
            self.max_num_instances = len(self.kb.thing.instances) - self.min_num_instances

        if self.search_algo == 'dfs':
            valid_concepts = self._apply_dfs()
        elif self.search_algo == 'strict-dfs':
            valid_concepts = self._apply_dfs(strict=True)
        else:
            print(f'Invalid input: search_algo:{search_algo} must be in [dfs,strict-dfs]')
            raise ValueError

        # Assign instances to concept chain of generated example
        for i in valid_concepts:
            for x in retrieve_concept_chain(i):
                x.concept.instances = self.kb._concept_generator.instance_retrieval(x.concept)
            yield i

    def _apply_dfs(self, strict=False) -> Generator:
        """
        Apply depth first search with backtracking to generate concepts.

        @return:
        """
        # instance_retrieval = self.kb.concept_generator.__instance_retrieval
        instance_retrieval = self.kb.instance_retrieval

        def f1(x):
            a = self.max_length >= len(x) >= self.min_length
            if not a:
                return a
            x.concept.instances = instance_retrieval(x.concept)
            b = self.max_num_instances >= len(x.concept.instances) >= self.min_num_instances
            return b

        def f2(x):
            return self.max_length >= len(x) >= self.min_length

        refinements = self.apply_rho(self.rho.getNode(self.kb.thing, root=True))
        if self.min_num_instances:
            constrain_func = f1
        else:
            constrain_func = f2

        valid_states_gate = set()
        while True:
            try:
                state = next(refinements)
            except StopIteration:
                print('All top concepts are refined.')
                break

            if constrain_func(state):
                valid_states_gate.add(state)
                yield state
                if strict:
                    if len(valid_states_gate) >= self.num_problems * self.num_diff_runs:
                        break

            temp_gate = set()
            for v in self._apply_dfs_on_state(state=state,
                                              apply_rho=self.apply_rho,
                                              constrain_func=constrain_func,
                                              depth=self.depth,
                                              patience_per_depth=(self.num_problems // 2)):
                if v not in valid_states_gate:
                    valid_states_gate.add(v)
                    temp_gate.add(v)
                    yield v
                    if strict:
                        if len(temp_gate) >= self.num_problems or (
                                len(valid_states_gate) >= self.num_problems * self.num_diff_runs):
                            break
            if strict:
                if len(valid_states_gate) >= self.num_problems * self.num_diff_runs:
                    break

        # sanity checking after the search.
        try:
            assert len(valid_states_gate) >= self.num_diff_runs * self.num_problems
        except AssertionError:
            print(f'Number of valid concepts generated:{len(valid_states_gate)}.\n'
                  f'Required number of concepts: {self.num_diff_runs * self.num_problems}.\n'
                  f'Please update the given constraints:'
                  f'Increase the max length (Currently {self.max_length}).\n'
                  f'Increase the max number of instances for concepts (Currently {self.max_num_instances}).\n'
                  f'Decrease the min length (Currently {self.min_length}).\n'
                  f'Decrease the max number of instances for concepts (Currently {self.min_num_instances}).\n')

    @staticmethod
    # @performance_debugger('_apply_dfs_on_state')
    def _apply_dfs_on_state(state, depth, apply_rho, constrain_func=None, patience_per_depth=None) -> set:
        """

        @param state:
        @param depth:
        @param apply_rho:
        @param num_problems:
        @param max_length:
        @param min_length:
        @param min_num_ind:
        @param max_num_ind:
        @return:
        """
        valid_examples = set()
        q = PriorityQueue()
        for _ in range(depth):
            temp_patience = patience_per_depth  # patience for valid exam. per depth.
            temp_not_valid_patience = patience_per_depth  # patience for not valid exam. per depth.
            for i in apply_rho(state):
                if constrain_func(i):  # validity checking.
                    # q.put((len(i), i))  # lower the length, higher priority.
                    if i not in valid_examples:
                        valid_examples.add(i)
                        q.put((len(i), i))  # lower the length, higher priority.
                        yield i
                        temp_patience -= 1
                        if temp_patience == 0:
                            break
                else:
                    # Heuristic if, too many of them not valid, do not continue.
                    temp_not_valid_patience -= 1
                    if temp_not_valid_patience == 0:
                        break

            if not q.empty():
                _, state = q.get()
            else:
                return None

    def apply_rho(self, node):
        for i in self.rho.refine(node):
            yield self.rho.getNode(i, parent_node=node)

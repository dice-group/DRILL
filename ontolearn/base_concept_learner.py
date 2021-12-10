from abc import ABCMeta, abstractmethod
from owlready2 import get_ontology
from .search import Node
from .metrics import F1, Accuracy
from typing import List, Set, Tuple, Dict, Any
from .util import create_experiment_folder, create_logger, get_full_iri
import numpy as np
import pandas as pd
import time
import random
import types
from .static_funcs import retrieve_concept_chain

import string
import sys


class BaseConceptLearner(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, knowledge_base=None, refinement_operator=None, heuristic_func=None, quality_func=None,
                 search_tree=None, max_num_of_concepts_tested=None, max_runtime=None, terminate_on_goal=None,
                 ignored_concepts=None, iter_bound=None, max_child_length=None, root_concept=None, verbose=None,
                 name=None):

        self.kb = knowledge_base
        self.rho = refinement_operator
        self.heuristic_func = heuristic_func
        self.quality_func = quality_func
        self.search_tree = search_tree
        self.max_num_of_concepts_tested = max_num_of_concepts_tested
        self.terminate_on_goal = terminate_on_goal
        self.max_runtime = max_runtime
        self.concepts_to_ignore = ignored_concepts
        self.iter_bound = iter_bound
        self.start_class = root_concept
        self.max_child_length = max_child_length
        self.verbose = verbose
        self.store_onto_flag = False
        self.start_time = None
        self.goal_found = False
        self.storage_path, _ = create_experiment_folder()
        self.name = name
        self.logger = create_logger(name=self.name, p=self.storage_path)
        self.last_path = None  # path of lastly stored onto.
        self.__default_values()
        self.__sanity_checking()

    def __default_values(self):
        """
        Fill all params with plausible default values.
        """

        self.search_tree.clean()
        self.search_tree.set_quality_func(self.quality_func)
        self.search_tree.set_heuristic_func(self.heuristic_func)

        if self.start_class is None:
            self.start_class = self.kb.thing
        if self.iter_bound is None:
            self.iter_bound = 10_000

        if self.max_num_of_concepts_tested is None:
            self.max_num_of_concepts_tested = 10_000
        if self.terminate_on_goal is None:
            self.terminate_on_goal = True
        if self.max_runtime is None:
            self.max_runtime = 3

        if self.max_child_length is None:
            self.max_child_length = 10

        if self.concepts_to_ignore is None:
            self.concepts_to_ignore = set()
        if self.verbose is None:
            self.verbose = 1

    def __sanity_checking(self):
        assert self.start_class
        assert self.search_tree is not None
        assert self.quality_func
        assert self.heuristic_func
        assert self.rho
        assert self.kb

        self.add_ignored_concepts(self.concepts_to_ignore)

    def add_ignored_concepts(self, ignore: Set[str]):
        """
        Ignore given set of concepts during search.
        Map set of strings concept e.g. 'Female', 'Mother' into our objects
        """
        if ignore:
            owl_concepts_to_ignore = set()
            for i in ignore:  # iterate over string representations of ALC concepts.
                found = False

                for IRI, concept in self.kb.uri_to_concept.items():
                    if (i == IRI) or (i == concept.name):
                        found = True
                        owl_concepts_to_ignore.add(concept)
                        break
                if found is False:
                    raise ValueError(
                        '{0} could not found in \n{1} \n{2}.'.format(i,
                                                                     [c.name for c in self.kb.uri_to_concept.values()],
                                                                     [uri for uri in self.kb.uri_to_concept.keys()]))
            self.concepts_to_ignore = owl_concepts_to_ignore  # use ALC concept representation instead of URI.

    def initialize_learning_problem(self, pos: Set[str], neg: Set[str], all_instances, ignore: Set[str]):
        """
        Determine the learning problem and initialize the search.
        1) Convert the string representation of an individuals into the owlready2 representation.
        2) Sample negative examples if necessary.
        3) Initialize the root and search tree.
        """
        self.default_state_concept_learner()

        assert len(self.kb.uri_to_concept) > 0

        assert isinstance(pos, set) and isinstance(neg, set) and isinstance(all_instances, set)
        assert 0 < len(pos) < len(all_instances) and len(all_instances) > len(neg)
        if self.verbose > 1:
            self.logger.info('E^+:[ {0} ]'.format(', '.join(pos)))
            self.logger.info('E^-:[ {0}] '.format(', '.join(neg)))
            if ignore:
                self.logger.info('Concepts to ignore:{0}'.format(' '.join(ignore)))
        self.add_ignored_concepts(ignore)
        unlabelled = all_instances.difference(pos.union(neg))
        self.quality_func.set_positive_examples(pos)
        self.quality_func.set_negative_examples(neg)

        self.heuristic_func.set_positive_examples(pos)
        self.heuristic_func.set_negative_examples(neg)
        self.heuristic_func.set_unlabelled_examples(unlabelled)

        root = self.rho.getNode(self.start_class, root=True)
        self.search_tree.quality_func.apply(root)
        self.search_tree.heuristic_func.apply(root)
        self.search_tree.add(root)
        assert len(self.search_tree) == 1

    def store_ontology(self):
        """

        @return:
        """
        # sanity checking.
        # (1) get all concepts
        # (2) serialize current kb.
        # (3) reload (2).
        # (4) get all reloaded concepts.
        # (5) (1) and (4) must be same.
        uri_all_concepts = set([get_full_iri(i) for i in self.kb.onto.classes()])
        self.last_path = self.storage_path + '/' + self.kb.name + str(time.time()) + '.owl'
        self.kb.save(path=self.last_path)  # save
        d = get_ontology(self.last_path).load()  # load it.
        uri_all_concepts_loaded = set([get_full_iri(i) for i in d.classes()])
        assert uri_all_concepts == uri_all_concepts_loaded

    def clean(self):
        self.concepts_to_ignore.clear()

    def train(self, *args, **kwargs):
        pass

    def terminate(self):
        """

        @return:
        """
        if self.store_onto_flag:
            self.store_ontology()

        if self.verbose == 1:
            self.logger.info('Elapsed runtime: {0} seconds'.format(round(time.time() - self.start_time, 4)))
            self.logger.info('Number of concepts tested:{0}'.format(self.number_of_tested_concepts))
            if self.goal_found:
                t = 'A goal concept found:{0}'.format(self.goal_found)
            else:
                t = 'Current best concept:{0}'.format(self.best_hypotheses(n=1)[0])
            self.logger.info(t)
            print(t)

        if self.verbose > 1:
            self.search_tree.show_search_tree('Final')

        self.clean()
        return self

    def get_metric_key(self, key: str):
        if key == 'quality':
            metric = self.quality_func.name
            attribute = key
        elif key == 'heuristic':
            metric = self.heuristic.name
            attribute = key
        elif key == 'length':
            metric = key
            attribute = key
        else:
            raise ValueError('Invalid key:{0}'.format(key))
        return metric, attribute

    @abstractmethod
    def next_node_to_expand(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply_rho(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @staticmethod
    def __prepare_str_target(target_concept):
        if isinstance(target_concept, Node):
            target_concept_str = target_concept.concept.name
        elif isinstance(target_concept, str):
            target_concept_str = target_concept
        else:
            raise ValueError
        return target_concept_str

    def fit_from_iterable(self, dataset: List, max_runtime: int = None) -> List[Dict]:
        if max_runtime:
            self.max_runtime = max_runtime

        results = []
        assert isinstance(dataset, List)
        for d in dataset:
            target_concept, positives, negatives, ignore_concepts = d['target_concept'], d['positive_examples'], d[
                'negative_examples'], d['ignore_concepts']

            target_concept_str = self.__prepare_str_target(target_concept)

            start_time = time.time()
            self.fit(pos=positives, neg=negatives, ignore=ignore_concepts)
            rn = time.time() - start_time
            if 'Drill' in self.name:
                # ADD KG processing time
                rn += self.time_kg_processing
            top_predictions = []
            for i in self.best_hypotheses():
                top_predictions.append([i.concept.name, f'Quality:{i.quality}'])
            h = self.best_hypothesis()
            individuals = h.concept.instances

            f_measure = F1().score(pos=positives, neg=negatives, instances=individuals)
            accuracy = Accuracy().score(pos=positives, neg=negatives, instances=individuals)
            report = {'Target': target_concept_str,
                      'Prediction': h.concept.name,
                      'TopPredictions': top_predictions,
                      'F-measure': f_measure,
                      'Accuracy': accuracy,
                      'NumClassTested': self.quality_func.num_times_applied,
                      'Runtime': rn}
            results.append(report)
        return results

    def best_hypotheses(self, n=10) -> List[Node]:
        assert self.search_tree is not None
        assert len(self.search_tree) > 1
        return [i for i in self.search_tree.get_top_n_nodes(n)]

    def best_hypothesis(self):
        return next(self.search_tree.get_top_n_nodes(1))

    @staticmethod
    def assign_labels_to_individuals(*, individuals: List, hypotheses: List[Node]) -> np.ndarray:
        """
        individuals: A list of owlready individuals.
        hypotheses: A

        Use each hypothesis as a binary function and assign 1 or 0 to each individual.

        return matrix of |individuals| x |hypotheses|
        """
        labels = np.zeros((len(individuals), len(hypotheses)))
        for ith_ind in range(len(individuals)):
            for jth_hypo in range(len(hypotheses)):
                if individuals[ith_ind] in hypotheses[jth_hypo].concept.instances:
                    labels[ith_ind][jth_hypo] = 1
        return labels

    @property
    def number_of_tested_concepts(self):
        return self.quality_func.applied

    def default_state_concept_learner(self):
        """
        At each problem initialization, we recent previous info if available.
        @return:
        """
        self.concepts_to_ignore.clear()
        self.search_tree.clean()
        self.quality_func.clean()
        self.heuristic_func.clean()

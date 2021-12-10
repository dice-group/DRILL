from typing import List, Tuple, Set, Dict, Any, Iterable
import numpy as np
import json
from sklearn.model_selection import KFold
from .abstracts import BaseNode
import time
from random import shuffle
import pandas as pd


class Experiments:
    def __init__(self, max_test_time_per_concept=3):
        self.random_state_k_fold = 1
        self.max_test_time_per_concept = max_test_time_per_concept

    @staticmethod
    def store_report(model, learning_problems: List[Iterable], test_report: List[dict]) -> Tuple[str, Dict[str, Any]]:
        """

        @param model: concept learner
        @param learning_problems: A list of learning problems (lps) where lp corresponds to [target concept, positive
        and negative examples, respectively.
        @param test_report: A list of predictions (preds) where
        test_report => { 'Prediction': str, 'F-measure': float, 'Accuracy', 'Runtime':float}
        @return:
        """

        assert len(learning_problems) == len(test_report)
        assert isinstance(learning_problems, list)  # and isinstance(learning_problems[0], list)
        assert isinstance(test_report, list) and isinstance(test_report[0], dict)

        store_json = dict()
        print('###############')
        for (th, lp, pred) in zip(range(len(learning_problems)), learning_problems, test_report):
            report = dict()
            # lp: {'target_concept':'..','positive_examples':set(),
            # 'negative_examples':set(),'ignore_concepts':set()}
            report['TargetConcept'] = lp['target_concept']
            """            
            # lp is a list
            # where lp[0] target node object or string
            # where lp[1] Positives
            # where lp[2] Negatives
            if hasattr(lp[0], 'concept'):
                report['TargetConcept'] = lp[0].concept.name
            else:
                report['TargetConcept'] = lp[0]
                print(f'Target => {report["TargetConcept"]}')
            """
            report.update(pred)
            report['Positives'], report['Negatives'] = list(lp['positive_examples']), list(
                lp['negative_examples'])  # 'set' is not JSON serializable.
            store_json[th] = report
        print('##################')
        # json serialize
        with open(f'{model.storage_path}/{model.name}_classification_reports.json', 'w') as file_descriptor:
            json.dump(store_json, file_descriptor, indent=3)

        del store_json

        # json serialize
        with open(f'{model.storage_path}/{model.name}_classification_reports.json', 'r') as read_file:
            report = json.load(read_file)
        array_res = np.array(
            [[v['F-measure'], v['Accuracy'], v['NumClassTested'], v['Runtime']] for k, v in report.items()])

        f1, acc, num_concept_tested, runtime = array_res[:, 0], array_res[:, 1], array_res[:, 2], array_res[:, 3]
        del array_res
        m = '{}\t F-measure:(avg.{:.2f} | std.{:.2f})\tAccuracy:(avg.{:.2f} | std.{:.2f})\t' \
            '\tNumClassTested:(avg.{:.2f} | std.{:.2f})\tRuntime:(avg.{:.2f} | std.{:.2f})'.format(model.name,
                                                                                                   f1.mean(), f1.std(),
                                                                                                   acc.mean(),
                                                                                                   acc.std(),
                                                                                                   num_concept_tested.mean(),
                                                                                                   num_concept_tested.std(),
                                                                                                   runtime.mean(),
                                                                                                   runtime.std())
        return m, {'F-measure': f1, 'Accuracy': acc, 'NumClassTested': num_concept_tested, 'Runtime': runtime}

    def start_KFold(self, k=None, dataset: List[Tuple[str, Set, Set]] = None, models: Iterable = None):
        """
        Perform KFold cross validation
        @param models:
        @param k:
        @param dataset: A list of tuples where a tuple (i,j,k) where i denotes the target concept
        j denotes the set of positive examples and k denotes the set of negative examples.
        @return:
        """
        models = {i for i in models}
        assert len(models) > 0
        assert len(dataset) > 0
        assert isinstance(dataset[0], tuple)
        assert isinstance(dataset[0], tuple)
        assert k
        dataset = np.array(dataset)  # due to indexing feature required in the sklearn.KFold.

        kf = KFold(n_splits=k, random_state=self.random_state_k_fold, shuffle=True)

        results = dict()
        counter = 1
        for train_index, test_index in kf.split(dataset):
            train, test = dataset[train_index].tolist(), dataset[test_index].tolist()
            print(f'##### FOLD:{counter} #####')
            start_time_fold = time.time()
            for m in models:
                m.train(train)
                test_report: List[dict] = m.fit_from_iterable(test, max_runtime=self.max_test_time_per_concept)
                str_report, dict_report = self.store_report(m, test, test_report)
                results.setdefault(m.name, []).append((counter, dict_report))
            print(f'##### FOLD:{counter} took {round(time.time() - start_time_fold)} seconds #####')
            counter += 1
        self.report_results(results)

    def start(self, dataset: List[Tuple[str, Set, Set]] = None, models: List = None):
        """
        Perform KFold cross validation
        @param models:
        @param k:
        @param dataset: A list of tuples where a tuple (i,j,k) where i denotes the target concept
        j denotes the set of positive examples and k denotes the set of negative examples.
        @return:
        """
        assert len(models) > 0
        assert len(dataset) > 0

        # shuffle(dataset)
        results = dict()
        counter = 1
        for m in models:
            print(
                f'{m.name} starts on {len(dataset)} number of problems. Max Runtime per problem is set to {self.max_test_time_per_concept} seconds.')
            test_report: List[dict] = m.fit_from_iterable(dataset, max_runtime=self.max_test_time_per_concept)
            str_report, dict_report = self.store_report(m, dataset, test_report)
            results.setdefault(m.name, []).append((counter, dict_report))
        # Save and ReportReport Results
        self.report_results(results, num_problems=len(dataset))

    @staticmethod
    def report_results(results: Dict[str, List], num_problems: int):
        print(f'\n##### RESULTS on {num_problems} number of learning problems#####')
        for learner_name, v in results.items():
            r = np.array([[report['F-measure'], report['Accuracy'], report['NumClassTested'], report['Runtime']] for
                          (fold, report) in v])
            f1_mean, f1_std = r[:, 0].mean(), r[:, 0].std()
            acc_mean, acc_std = r[:, 1].mean(), r[:, 1].std()
            num_concept_tested_mean, num_concept_tested_std = r[:, 2].mean(), r[:, 2].std()

            runtime_mean, runtime_std = r[:, 3].mean(), r[:, 3].std()

            print(
                f'{learner_name}\t F-measure:(avg. {f1_mean:.2f} | std. {f1_std:.2f})\tAccuracy:(avg. {acc_mean:.2f} | std. {acc_std:.2f})\t\tNumClassTested:(avg. {num_concept_tested_mean:.2f} | std. {num_concept_tested_std:.2f})\t\tRuntime:(avg.{runtime_mean:.2f} | std.{runtime_std:.2f})')

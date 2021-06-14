"""
====================================================================
Drill -- Deep Reinforcement Learning for Refinement Operators in ALC
====================================================================
Reproducing our experiments Experiments

This script performs the following computations
1. Parse KG.
2. Load learning problems LP= {(E^+,E^-)...]

3. Initialize models .
    3.1. Initialize DL-learnerBinder objects to communicate with DL-learner binaries.
    3.2. Initialize DRILL.
4. Provide models + LP to Experiments object.
    4.1. Each learning problem provided into models
    4.2. Best hypothesis/predictions of models given E^+ and E^- are obtained.
    4.3. F1-score, Accuracy, Runtimes and Number description tested information stored and serialized.
"""
from ontolearn import KnowledgeBase, LearningProblemGenerator, DrillAverage
from ontolearn import Experiments
from ontolearn.binders import DLLearnerBinder
import pandas as pd
from argparse import ArgumentParser
import os
import json
import time

full_computation_time = time.time()


def sanity_checking_args(args):
    try:
        assert os.path.isfile(args.path_knowledge_base)
    except AssertionError as e:
        print(f'--path_knowledge_base ***{args.path_knowledge_base}*** does not lead to a file.')
        exit(1)
    assert os.path.isfile(args.path_knowledge_base_embeddings)
    assert os.path.isfile(args.path_knowledge_base)


def start(args):
    sanity_checking_args(args)
    kb = KnowledgeBase(args.path_knowledge_base)
    with open(args.path_lp) as json_file:
        settings = json.load(json_file)
    problems = [(k, set(v['positive_examples']), set(v['negative_examples'])) for k, v in
                settings['problems'].items()]

    print(f'Number of problems {len(problems)} on {kb}')

    # Initialize models
    celoe = DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='celoe')
    ocel = DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='ocel')
    eltl = DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='eltl')

    drill_average = DrillAverage(pretrained_model_path=args.pretrained_drill_avg_path, knowledge_base=kb,
                                 path_of_embeddings=args.path_knowledge_base_embeddings,
                                 verbose=args.verbose, num_workers=args.num_workers)

    print(f'KG preprocessing took : {time.time() - full_computation_time}')
    Experiments(max_test_time_per_concept=args.max_test_time_per_concept).start(dataset=problems,
                                                                                models=[
                                                                                    drill_average,ocel, celoe, eltl])


if __name__ == '__main__':
    parser = ArgumentParser()
    # LP dependent
    parser.add_argument("--path_knowledge_base", type=str)
    parser.add_argument("--path_lp", type=str)
    parser.add_argument("--path_knowledge_base_embeddings", type=str)
    parser.add_argument('--pretrained_drill_avg_path', type=str, help='Provide a path of .pth file')
    # Binaries for DL-learner
    parser.add_argument("--path_dl_learner", type=str)
    # Concept Learning Testing
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3, help='Max. runtime during testing')

    # General
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during batching')

    start(parser.parse_args())

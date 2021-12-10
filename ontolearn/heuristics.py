from .abstracts import AbstractScorer
import numpy as np

class Reward(AbstractScorer):
    """
    A reward function based on the CELOE heuristic function.
    """
    def __init__(self, pos=None, neg=None, unlabelled=None, reward_of_goal=5.0, beta=.04, alpha=.5):
        super().__init__(pos, neg, unlabelled)
        self.name = 'F1'

        self.reward_of_goal = reward_of_goal
        self.beta = beta
        self.alpha = alpha

    def score(self, pos, neg, instances):
        self.pos = pos
        self.neg = neg

        tp = len(self.pos.intersection(instances))
        tn = len(self.neg.difference(instances))

        fp = len(self.neg.intersection(instances))
        fn = len(self.pos.difference(instances))
        try:
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f_1 = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f_1 = 0

        return round(f_1, 5)

    def apply(self, node):
        """
        Calculate F1-score and assigns it into quality variable of node.
        """
        self.applied += 1

        instances = node.concept.instances
        if len(instances) == 0:
            node.quality = 0
            return False

        tp = len(self.pos.intersection(instances))
        # tn = len(self.neg.difference(instances))

        fp = len(self.neg.intersection(instances))
        fn = len(self.pos.difference(instances))

        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            node.quality = 0
            return False

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            node.quality = 0
            return False

        if precision == 0 or recall == 0:
            node.quality = 0
            return False

        f_1 = 2 * ((precision * recall) / (precision + recall))
        node.quality = round(f_1, 5)

        assert node.quality

    def calculate(self, current_state, next_state=None) -> float:
        self.apply(current_state)
        self.apply(next_state)
        if next_state.quality == 1.0:
            return self.reward_of_goal

        reward = next_state.quality
        # Reward => being better than parent.
        if next_state.quality > current_state.quality:
            reward += (next_state.quality - current_state.quality) * self.alpha

        # Regret => Length penalization.
        reward -= len(next_state) * self.beta

        return max(reward, 0)

class BinaryReward(AbstractScorer):
    """
    Receive reward only if you reach a goal state
    """
    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'BinaryReward'

    def score(self, pos, neg, instances):
        self.pos = pos
        self.neg = neg

        tp = len(self.pos.intersection(instances))
        tn = len(self.neg.difference(instances))

        fp = len(self.neg.intersection(instances))
        fn = len(self.pos.difference(instances))
        try:
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f_1 = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f_1 = 0

        return round(f_1, 5)

    def apply(self, node):
        """
        Calculate F1-score and assigns it into quality variable of node.
        """
        self.applied += 1

        instances = node.concept.instances
        if len(instances) == 0:
            node.quality = 0
            return False

        tp = len(self.pos.intersection(instances))
        # tn = len(self.neg.difference(instances))

        fp = len(self.neg.intersection(instances))
        fn = len(self.pos.difference(instances))

        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            node.quality = 0
            return False

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            node.quality = 0
            return False

        if precision == 0 or recall == 0:
            node.quality = 0
            return False

        f_1 = 2 * ((precision * recall) / (precision + recall))
        node.quality = round(f_1, 5)

        assert node.quality

    def calculate(self, current_state, next_state=None) -> float:
        self.apply(current_state)
        # TODO: should not current_state satisfy following constraints ?
        # assert isinstance(current_state.quality,float)
        # assert current_state.quality >= 1
        self.apply(next_state)
        if next_state.quality == 1.0:
            return 1.0
        else:
            return 0

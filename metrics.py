from typing import Set
from abstracts import AbstractScorer


class Recall(AbstractScorer):
    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'Recall'

    def score(self, pos, neg, instances):
        self.pos = pos
        self.neg = neg
        if len(instances) == 0:
            return 0
        tp = len(self.pos.intersection(instances))
        fn = len(self.pos.difference(instances))
        try:
            recall = tp / (tp + fn)
            return round(recall, 5)
        except ValueError:
            return 0

    def apply(self, node):
        self.applied += 1

        instances = node.concept.instances
        if len(instances) == 0:
            node.quality = 0
            return False
        tp = len(self.pos.intersection(instances))
        fn = len(self.pos.difference(instances))
        try:
            recall = tp / (tp + fn)
            node.quality = round(recall, 5)
        except ZeroDivisionError:
            node.quality = 0
            return False


class Precision(AbstractScorer):
    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'Precision'

    def score(self, pos, neg, instances):
        self.pos = pos
        self.neg = neg
        if len(instances) == 0:
            return 0
        tp = len(self.pos.intersection(instances))
        fp = len(self.neg.intersection(instances))
        try:
            precision = tp / (tp + fp)
            return round(precision, 5)
        except ValueError:
            return 0

    def apply(self, node):
        self.applied += 1
        instances = node.concept.instances
        if len(instances) == 0:
            node.quality = 0
            return False
        tp = len(self.pos.intersection(instances))
        fp = len(self.neg.intersection(instances))
        try:
            precision = tp / (tp + fp)
            node.quality = round(precision, 5)
        except ZeroDivisionError:
            node.quality = 0
            return False


class F1(AbstractScorer):
    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'F1'
        self.beta = 0
        self.noise = 0

    def __call__(self, pos, neg, individuals):
        assert isinstance(pos,set) and len(pos)>0
        assert isinstance(neg, set) and len(pos)>0
        assert isinstance(individuals, set) and len(pos)>0

        return self.score(pos, neg, individuals)

    def score(self, pos:Set[str], neg:Set[str], instances:Set[str]):
        self.pos = pos
        self.neg = neg

        # TRUE POSITIVES : #. of overlap btw E^+ and instances ( higher the better)
        tp = len(self.pos.intersection(instances))

        # TRUE NEGATIVES : #. of diff. btw E^- and instances ( higher the better)
        # tn = len(self.neg.difference(instances))

        # FALSE POSITIVES: #. of overlap btw E^- and instances ( lower the better)
        fp = len(self.neg.intersection(instances))

        # FALSE NEGATIVES: #. of diff. btw E^+ and instances ( lower the better)
        fn = len(self.pos.difference(instances))
        try:
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f_1 = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f_1 = 0.0

        return round(f_1, 5)

    def apply(self, node):
        self.applied += 1

        instances = node.concept.instances
        if len(instances) == 0:
            node.quality = 0
            return False

        # TRUE POSITIVES : #. of overlap btw E^+ and instances ( higher the better)
        tp = len(self.pos.intersection(instances))

        # TRUE NEGATIVES : #. of diff. btw E^- and instances ( higher the better)
        # tn = len(self.neg.difference(instances))

        # FALSE POSITIVES: #. of overlap btw E^- and instances ( lower the better)
        fp = len(self.neg.intersection(instances))

        # FALSE NEGATIVES: #. of diff. btw E^+ and instances ( lower the better)
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
        node.quality = f_1  # round(f_1, 5)

        assert node.quality


class Accuracy(AbstractScorer):
    def score(self, pos, neg, instances):
        self.pos = pos
        self.neg = neg

        # TRUE POSITIVES : #. of overlap btw E^+ and instances ( higher the better)
        tp = len(self.pos.intersection(instances))

        # TRUE NEGATIVES : #. of diff. btw E^- and instances ( higher the better)
        tn = len(self.neg.difference(instances))

        # FALSE POSITIVES: #. of overlap btw E^- and instances ( lower the better)
        fp = len(self.neg.intersection(instances))

        # FALSE NEGATIVES: #. of diff. btw E^+ and instances ( lower the better)
        fn = len(self.pos.difference(instances))

        try:
            acc = (tp + tn) / (tp + tn + fp + fn)
        except ZeroDivisionError as e:
            print(e)
            print(tp)
            print(tn)
            print(fp)
            print(fn)
            acc = 0
        return acc

    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'Accuracy'

    def apply(self, node):
        self.applied += 1

        instances = node.concept.instances
        if len(instances) == 0:
            node.quality = 0
            return False

        # TRUE POSITIVES : #. of overlap btw E^+ and instances ( higher the better)
        tp = len(self.pos.intersection(instances))

        # TRUE NEGATIVES : #. of diff. btw E^- and instances ( higher the better)
        tn = len(self.neg.difference(instances))

        # FALSE POSITIVES: #. of overlap btw E^- and instances ( lower the better)
        fp = len(self.neg.intersection(instances))

        # FALSE NEGATIVES: #. of diff. btw E^+ and instances ( lower the better)
        fn = len(self.pos.difference(instances))

        acc = (tp + tn) / (tp + tn + fp + fn)
        # acc = 1 - ((fp + fn) / len(self.pos) + len(self.neg)) # from Learning OWL Class Expressions.

        node.quality = acc  # round(acc, 5)

from abc import ABC,abstractmethod

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
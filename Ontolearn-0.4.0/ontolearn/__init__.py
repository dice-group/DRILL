"""Structured Machine learning modules for Python

Ontolearn is an open-source software library for structured machine learning in Python
The goal of ontolearn is to provide efficient solutions for concept learning on RDF knowledge bases


Author:
    The Ontolearn team <onto-learn@lists.uni-paderborn.de>
"""
__version__ = '0.4.0'

from .knowledge_base import KnowledgeBase
from .abstracts import *
from .base_concept_learner import *
from .metrics import *
from .search import *
__all__ = 'KnowledgeBase',

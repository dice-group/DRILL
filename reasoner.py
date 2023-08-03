# from util import jaccard_similarity, compute_prediction, evaluate_results
import concurrent.futures
import dicee
import requests
from abc import ABC, abstractmethod
from typing import Set, Iterable, Tuple, Callable, Dict
from owlapy.parser import DLSyntaxParser
from owlapy.owl2sparql.converter import Owl2SparqlConverter
import time
import asyncio

"""
class AbstractDLConcept(ABC):
    pass


class Restriction(AbstractDLConcept):
    def __init__(self, opt: str = None, role: str = None, filler=None):
        super(Restriction, self)
        assert opt == '∃' or opt == '∀'
        assert role is not None
        self.opt = opt
        self.role_iri = role
        self.role = role.split('#')[-1][:-1]
        self.filler = filler
        if self.filler.length == 1:
            self.str = self.opt + ' ' + self.role + '.' + filler.str
        else:
            self.str = self.opt + ' ' + self.role + '.' + '(' + filler.str + ')'

        self.length = filler.length + 2

    @property
    def manchester_str(self):
        if self.opt == '∃':
            if self.filler.length > 1:
                return self.role + ' SOME ' + '(' + self.filler.manchester_str + ')'
            elif self.filler.length == 1:
                return self.role + ' SOME ' + self.filler.manchester_str
            else:
                raise KeyError(f'The length of the filler is invalid {self.filler.length}')
        elif self.opt == '∀':
            if self.filler.length > 1:
                return self.role + ' ONLY ' + '(' + self.filler.manchester_str + ')'
            elif self.filler.length == 1:
                return self.role + ' ONLY ' + self.filler.manchester_str
            else:
                raise KeyError(f'The length of the filler is invalid {self.filler.length}')
        else:
            raise KeyError(f'Invalid Opt. {self.opt}')

    def neg(self):
        if self.opt == '∃':
            # negation can be shifted past quantifiers.
            # ¬∃ r. C is converted into ∀ r. ¬C at the object creation
            # ¬∃ r. C \equiv ∀ r. ¬C
            return Restriction(opt='∀', role=self.role_iri, filler=self.filler.neg())
        elif self.opt == '∀':
            # ¬∀ r. C \equiv ¬∃ r. C
            return Restriction(opt='∃', role=self.role_iri, filler=self.filler.neg())
        else:
            raise KeyError(f'Wrong opt {self.opt}')

    def union(self, other):
        return DisjunctionDLConcept(concept_a=self, concept_b=other)

    def intersection(self, other):
        return ConjunctionDLConcept(concept_a=self, concept_b=other)


class ValueRestriction(AbstractDLConcept):
    def __init__(self, opt: str = None, val: int = None, role: str = None, filler=None):
        super(ValueRestriction, self)
        assert opt == '≥' or opt == '≤'
        assert role is not None
        assert isinstance(val, int)
        self.opt = opt
        self.val = val
        self.role_iri = role
        self.role = role.split('#')[-1][:-1]
        self.filler = filler
        self.str = self.opt + ' ' + f'{self.val} ' + self.role + '.' + filler.str
        self.sparql = None

    @property
    def manchester_str(self):
        if self.opt == '≥':
            return self.role + ' MIN ' + f'{self.val} ' + self.filler.manchester_str
        elif self.opt == '≤':
            return self.role + ' MAX ' + f'{self.val} ' + self.filler.manchester_str


class ConjunctionDLConcept(AbstractDLConcept):
    def __init__(self, concept_a, concept_b):
        super(ConjunctionDLConcept, self)
        if concept_a.length == 1 and concept_b.length == 1:
            self.str = concept_a.str + " ⊓ " + concept_b.str
        elif concept_a.length == 1 and concept_b.length > 1:
            self.str = concept_a.str + " ⊓ " + "(" + concept_b.str + ")"
        elif concept_a.length > 1 and concept_b.length == 1:
            # Put the shorted one first.
            self.str = concept_b.str + " ⊓ " + "(" + concept_a.str + ")"
        elif concept_a.length > 1 and concept_b.length > 1:
            self.str = "(" + concept_a.str + " ⊓ " + concept_b.str + ")"
        else:
            raise KeyError()
        self.left = concept_a
        self.right = concept_b
        self.length = concept_a.length + concept_b.length + 1
        self.iri = [(self.left.iri, self.right.iri)]

    @property
    def manchester_str(self):
        left_part = self.left.manchester_str
        right_part = self.right.manchester_str
        if self.left.length > 1:
            left_part = "(" + left_part + ")"

        if self.right.length > 1:
            right_part = "(" + right_part + ")"
        return left_part + " AND " + right_part

    def neg(self):
        # \neg (C \sqcup D) \equiv \neg C \sqcap \neg D
        return DisjunctionDLConcept(concept_a=self.left.neg(), concept_b=self.right.neg())

    def union(self, other: AbstractDLConcept):
        return DisjunctionDLConcept(concept_a=self, concept_b=other)

    def intersection(self, other: AbstractDLConcept):
        return ConjunctionDLConcept(concept_a=self, concept_b=other)


class DisjunctionDLConcept(AbstractDLConcept):
    def __init__(self, concept_a, concept_b):
        super(DisjunctionDLConcept, self)
        if concept_a.length == 1 and concept_b.length == 1:
            self.str = concept_a.str + " ⊔ " + concept_b.str
        elif concept_a.length == 1 and concept_b.length > 1:
            self.str = concept_a.str + " ⊔ " + concept_b.str
        elif concept_b.length == 1 and concept_a.length > 1:
            self.str = concept_b.str + " ⊔ " + concept_a.str
        elif concept_b.length > 1 and concept_a.length > 1:
            self.str = "(" + concept_a.str + " ⊔ " + concept_b.str + ")"
        else:
            raise KeyError()

        self.left = concept_a
        self.right = concept_b
        self.length = concept_a.length + concept_b.length + 1
        self.iri = [(self.left.iri, self.right.iri)]

    @property
    def manchester_str(self):
        left_part = self.left.manchester_str
        right_part = self.right.manchester_str
        if self.left.length > 1:
            left_part = "(" + left_part + ")"

        if self.right.length > 1:
            right_part = "(" + right_part + ")"
        return left_part + " OR " + right_part

    def neg(self):
        # \neg (C \sqcap D) \equiv \neg C \sqcup \neg D
        return ConjunctionDLConcept(concept_a=self.left.neg(), concept_b=self.right.neg())

    def union(self, other: AbstractDLConcept):
        return DisjunctionDLConcept(concept_a=self, concept_b=other)

    def intersection(self, other: AbstractDLConcept):
        return ConjunctionDLConcept(concept_a=self, concept_b=other)


class NC(AbstractDLConcept):
    def __init__(self, iri: str):
        super(NC, self).__init__()
        assert isinstance(iri, str)
        assert len(iri) > 0
        self.iri = iri
        assert self.iri[0] == '<' and self.iri[-1] == '>'
        self.str = self.iri.split('#')[-1][:-1]
        self.length = 1

    @property
    def manchester_str(self):
        return self.str

    def neg(self):
        return NNC(iri=self.iri)

    def union(self, other: AbstractDLConcept):
        return DisjunctionDLConcept(self, other)

    def intersection(self, other: AbstractDLConcept):
        return ConjunctionDLConcept(self, other)


class NNC(AbstractDLConcept):
    def __init__(self, iri: str):
        super(NNC, self)
        assert iri[0] == '<' and iri[-1] == '>'
        self.neg_iri = iri
        self.str = "¬" + iri.split('#')[-1][:-1]
        self.length = 2

    @property
    def manchester_str(self):
        return self.str.replace('¬', 'NOT ')

    def neg(self):
        return NC(self.neg_iri)

    def union(self, other: AbstractDLConcept):
        return DisjunctionDLConcept(self, other)

    def intersection(self, other: AbstractDLConcept):
        return ConjunctionDLConcept(self, other)


class Thing:
    def __init__(self):
        self.str = '⊤'
        self.length = 1
        self.iri = "<http://www.w3.org/2002/07/owl#Thing>"

    @property
    def manchester_str(self):
        return "Thing"


class Nothing:
    def __init__(self):
        self.str = '⊥'
        self.length = 1
        self.iri = "<http://www.w3.org/2002/07/owl#Nothing>"

    @property
    def manchester_str(self):
        return "Nothing"

"""

from owlapy.model import OWLThing, OWLNothing, OWLClass, IRI, OWLObjectUnionOf, OWLObjectIntersectionOf


def named_concepts_and_base_iri(query: Callable) -> Tuple[Set[str], str]:
    # (1) Get All named concepts.
    iri_named_concepts = query("PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
                               "SELECT DISTINCT ?var\n"
                               "WHERE {?var a owl:Class.}")
    # (2) Retrieve the base iri of an input knowledge base.
    namespace = None
    # How do you get base IRI   ?
    for i in iri_named_concepts:
        if namespace is None:
            # start 1 because of the left bracket <...>
            namespace = i[1:i.index('#') + 1]
        else:
            assert namespace == i[1:i.index('#') + 1]
    return iri_named_concepts, namespace


def named_concepts(iri_named_concepts: Set[str], namespace: str) -> Tuple[Set[OWLClass], Dict[str, OWLClass]]:
    result = set()
    str_to_concept_mapping = dict()
    for i in iri_named_concepts:
        remainder = i[i.index(namespace) + len(namespace):-1]
        c = OWLClass(iri=IRI(namespace=namespace, remainder=remainder))
        result.add(c)
        if '<' + c.get_iri().as_str() + '>' in str_to_concept_mapping:
            raise RuntimeError('SPARQL Query did not correctly return distinc named concepts')
        str_to_concept_mapping['<' + c.get_iri().as_str() + '>'] = c
    return result, str_to_concept_mapping


def is_knowledge_base_materialized(query):
    str_individuals = query("PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
                            "SELECT DISTINCT ?var\n"
                            "WHERE {?var a owl:NamedIndividual.}")
    flag = str_individuals == query("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                                    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
                                    "SELECT DISTINCT ?var\n"
                                    "WHERE {\n"
                                    "?var a ?type.\n"
                                    "?type rdfs:subClassOf ?y.\n"
                                    "FILTER NOT EXISTS {?type rdf:type rdf:Class}\n"
                                    "FILTER NOT EXISTS {?type rdf:type rdf:Property}}")

    return flag, str_individuals


def get_roles(query, name_space):
    return {i for i in query("SELECT DISTINCT ?var\n"
                             "WHERE {?s ?var ?o.}") if name_space in i}


class SPARQLCWR:
    """
    A vocabulary [signature]
        1. containing individual names [constants],
        2. concept names [unary predicates]
        3. and role names [binary predicates].


        A named individual corresponds to (1) that is explicitly defined in KB.
        A named concept corresponds (2) that is explicitly defined in KB.


Two specific class names, > and ⊥, denote the concept containing all individuals and the empty concept, respectively
    """

    def __init__(self, url, name: str = 'sparqlcwr'):
        self.url = url
        self.name = name
        self.thing = OWLThing
        self.nothing = OWLNothing
        self.str_concepts, self.namespace = named_concepts_and_base_iri(query=self.query)
        self.concepts, self.str_to_named_concepts = named_concepts(self.str_concepts, self.namespace)
        self.materialized, self.str_individuals = is_knowledge_base_materialized(self.query)
        self.str_roles = get_roles(self.query, name_space=self.namespace)
        self.parser = DLSyntaxParser(self.namespace)
        self.converter = Owl2SparqlConverter()

    def __str__(self):
        return f"SPARQLCWR:|Individuals|:{len(self.named_individuals)}\t |Concepts|:{len(self.concepts)}"

    @property
    def individuals(self):
        return self.str_individuals

    def sparql(self, cls: OWLClass):
        return self.converter.as_query("?var", cls, False)

    def query(self, query: str) -> Set[str]:
        """
        Perform a SPARQL query
        :param query:
        :return:
        """
        response = requests.post(self.url, data={'query': query})
        if response.ok:
            answer = response.json()['results']['bindings']
            return {'<' + i['var']['value'] + '>' for i in answer}
        else:
            print(response)
            print(query)
            raise RuntimeWarning('Something went wrong..')

    def retrieve(self, concept) -> Set[str]:
        """
        perform concept retrieval
        :param concept:
        :return:
        """
        try:
            sparql_mapping = self.sparql(concept)
        except:
            print(concept)
            print(type(concept))
            print(isinstance(concept, OWLClass))
            exit(1)
            raise RuntimeError()

        return self.query(sparql_mapping)

    def apply_construction_rules(self, concept: OWLClass):
        results = set()
        for i in self.named_concepts:
            results.add(OWLObjectUnionOf(operands=[i, concept]))
            results.add(OWLObjectIntersectionOf(operands=[i, concept]))
        return results

    def __named_class_to_str(self, concept: OWLClass):
        assert concept in self.concepts
        return '<' + concept.get_iri().as_str() + '>'

    def construct_intersection(self, c, d):
        operands=[]
        if isinstance(c, OWLObjectIntersectionOf):
            operands.extend([i for i in c.operands()])
        else:
            operands.append(c)
        if isinstance(d, OWLObjectIntersectionOf):
            operands.extend([i for i in d.operands()])
        else:
            operands.append(d)
        return OWLObjectIntersectionOf(operands=operands)

    def refine(self, concept: OWLClass):
        result = set()
        # (1) Base for OWLThing
        if concept is OWLThing:
            str_direct_subclasses = self.query("PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
                                               "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                                               "SELECT DISTINCT ?var WHERE { owl:Thing  ^rdfs:subClassOf ?var .}")
            direct_subclasses = {self.str_to_named_concepts[i] for i in str_direct_subclasses}
            for i in direct_subclasses:
                result.add(i)
                result.add(self.construct_intersection(i, concept))
        # (2) Base for Named Concepts
        elif concept in self.concepts:

            str_direct_subclasses = self.query("PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
                                               "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                                               "SELECT DISTINCT ?var WHERE{\n"
                                               f"{self.__named_class_to_str(concept)}  ^rdfs:subClassOf ?var .\n"
                                               "}")
            direct_subclasses = {self.str_to_named_concepts[i] for i in str_direct_subclasses}
            for i in direct_subclasses:
                result.add(i)
                result.add(self.construct_intersection(i, concept))
            return result
        # (3)Base for Intersection
        elif isinstance(concept, OWLObjectIntersectionOf):
            # (3.1) Iterate over operands of C
            for c in concept.operands():
                # (3.2) Refine each (3.1)
                for i in self.refine(c):
                    # (3.3) Construct OWl Intersect with the given concept
                    result.add(self.construct_intersection(i, concept))
                    # TODO: we can remove the c from concept before create owl intersect
            result.add(self.construct_intersection(OWLThing, concept))
        else:
            raise RuntimeError('Unrecognized Type.')
        return result

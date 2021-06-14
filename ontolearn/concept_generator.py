import types
from owlready2 import Not, AllDisjoint
from .concept import Concept
import concurrent.futures
import owlready2
from typing import Dict
from .static_funcs import concepts_sorter, retrieve_concept_chain
from .util import get_full_iri

all_instances = None


def instances_of_universal_restriction(role_relations, concept_instances):
    temp = set()
    for a, b in role_relations:  # (a,b) \in r^I
        if not (b in concept_instances):  # b \in C^I
            temp.add(a)
    return all_instances - temp


def instances_of_existential_restriction(role_relations, concept_instances):
    temp = set()
    for a, b in role_relations:  # (a,b) \in r^I
        if b in concept_instances:
            temp.add(a)
    return temp


def instance_retrieval(concept):
    if concept.is_atomic:
        # We must have instance of atomic classes
        return concept.instances
    elif concept.form == 'ObjectComplementOf':
        global all_instances
        return all_instances - instance_retrieval(concept.concept_a)
    elif concept.form == 'ObjectUnionOf':
        return instance_retrieval(concept.concept_a) | instance_retrieval(concept.concept_b)
    elif concept.form == 'ObjectIntersectionOf':
        return instance_retrieval(concept.concept_a) & instance_retrieval(concept.concept_b)
    elif concept.form == 'ObjectSomeValuesFrom':
        instances_of_filler = instance_retrieval(concept.filler)
        return instances_of_existential_restriction(concept.role.relations, instances_of_filler)
    elif concept.form == 'ObjectAllValuesFrom':
        instances_of_filler = instance_retrieval(concept.filler)
        return instances_of_universal_restriction(concept.role.relations, instances_of_filler)
    else:
        raise ValueError


class ConceptGenerator:
    def __init__(self, concepts: Dict, thing: Concept, nothing: Concept, onto):

        self.uri_to_concepts = concepts
        self.thing = thing
        self.nothing = nothing
        self.onto = onto
        global all_instances
        all_instances = self.thing.instances

    @staticmethod
    def instance_retrieval(concept):
        return instance_retrieval(concept)

    @staticmethod
    def instance_retrieval_node(node, type_container=list):
        return type_container(instance_retrieval(node.concept))

    @staticmethod
    def instances_of_universal_restriction(role_relations, concept_instances):
        temp = set()
        for a, b in role_relations:  # (a,b) \in r^I
            if not (b in concept_instances):  # b \in C^I
                temp.add(a)
        global all_instances
        return all_instances - temp

    @staticmethod
    def instances_of_existential_restriction(role_relations, concept_instances):
        temp = set()
        for a, b in role_relations:
            if b in concept_instances:
                temp.add(a)
        return temp

    def negation(self, concept: Concept) -> Concept:
        # 2.
        if concept.name != '⊤' and concept.name != '⊥' and concept.form != 'ObjectComplementOf':

            c = Concept(name="¬{0}".format(concept.name),
                        instances=None,
                        form='ObjectComplementOf', concept_a=concept)
            return c
        # 3.
        elif concept.form == 'ObjectComplementOf':
            assert concept.name[0] == '¬'
            return concept.concept_a
        elif concept.name == 'Thing':
            return self.nothing
        elif concept.name == 'Nothing':
            return self.thing

        else:
            raise ValueError

    @staticmethod
    def existential_restriction(concept: Concept, relation) -> Concept:
        c = Concept(name="(∃{0}.{1})".format(relation.name, concept.name),
                    form='ObjectSomeValuesFrom',
                    instances=None,
                    role=relation, filler=concept)
        return c

    @staticmethod
    def universal_restriction(concept: Concept, relation) -> Concept:
        c = Concept(name="(∀{0}.{1})".format(relation.name, concept.name),
                    form='ObjectAllValuesFrom',
                    instances=None,
                    role=relation, filler=concept)
        return c

    @staticmethod
    def union(A: Concept, B: Concept):

        if A.name == '⊤' and B.name == '⊤':
            return A
        if A.name == '⊥' and B.name == '⊥':
            return A

        A, B = concepts_sorter(A, B)
        c = Concept(name="({0} ⊔ {1})".format(A.name, B.name),
                    instances=None,
                    form='ObjectUnionOf', concept_a=A, concept_b=B)
        return c

    @staticmethod
    def intersection(A: Concept, B: Concept) -> Concept:
        if A.name == '⊤' and B.name == '⊤':
            return A
        if A.name == '⊥' and B.name == '⊥':
            return A

        A, B = concepts_sorter(A, B)
        c = Concept(name="({0} ⊓ {1})".format(A.name, B.name),
                    instances=None,
                    form='ObjectIntersectionOf', concept_a=A, concept_b=B)
        return c

    def get_instances_for_restrictions(self, exist, role, filler):
        temp = set()
        if exist:
            for a, b in role.relations:  # (a,b) \in r^I
                if b in filler.instances:
                    temp.add(a)
            return temp
        else:
            for a, b in role.relations:  # (a,b) \in r^I
                if not (b in filler.instances):  # b \in C^I
                    temp.add(a)
            return self.thing.instances - temp

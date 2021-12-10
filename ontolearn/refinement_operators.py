from .base import KnowledgeBase
from .util import parametrized_performance_debugger
from .concept import Concept
from .search import Node
from .abstracts import BaseRefinement
import copy, random
from typing import Set, Generator, Iterable
from itertools import chain, tee


class LengthBasedRefinement(BaseRefinement):
    """ A top down refinement operator refinement operator in ALC."""

    def __init__(self, kb):
        super().__init__(kb)
        if 150 > len(self.kb.concepts) > 100:
            self.max_len_refinement_top = 2
        elif 100 > len(self.kb.concepts) > 50:
            self.max_len_refinement_top = 3
        else:
            self.max_len_refinement_top = 4
        self.top_refinements = None
        self.compute_top_refinements()

    def compute_top_refinements(self):
        self.top_refinements = []
        for ref in self.refine_top_concept():
            if ref.name != '⊥':
                self.top_refinements.append(ref)

    def remove_from_top_refinements(self, i: Concept):
        list_to_remove = []
        for j in self.top_refinements:
            if i == j:
                list_to_remove.append(i)
                continue
            if j.form in ['ObjectIntersectionOf', 'ObjectUnionOf']:
                if j.concept_a == i or j.concept_b == i:
                    list_to_remove.append(j)
            elif j.form in ['ObjectSomeValuesFrom', 'ObjectAllValuesFrom']:
                if j.filler == i:
                    list_to_remove.append(j)
            else:
                continue
        for i in list_to_remove:
            self.top_refinements.remove(i)

    def apply_union_and_intersection_from_iterable(self, cont: Iterable[Generator], ignore_union=False) -> Iterable:
        cumulative_refinements = dict()
        # 1. Flatten list of generators.
        for concept in chain.from_iterable(cont):
            if concept.name != '⊥':
                # 1.2. Store qualifying concepts based on their lengths.
                cumulative_refinements.setdefault(len(concept), set()).add(concept)
            else:
                """ No need to union or intersect Nothing, i.e. ignore concept that does not satisfy constraint"""
                yield concept

        # 2. Lengths of qualifying concepts.
        lengths = [i for i in cumulative_refinements.keys()]

        seen = set()
        larger_cumulative_refinements = dict()
        # 3. Iterative over lengths.
        for i in lengths:  # type: int
            # 3.1 return all i.th length
            yield from cumulative_refinements[i]
            for j in lengths:
                if (i, j) in seen or (j, i) in seen:
                    continue
                seen.add((i, j))
                seen.add((j, i))

                len_ = i + j + 1

                if len_ <= self.max_len_refinement_top:
                    # 3.2 Intersect concepts having length i with concepts having length j.
                    intersect_of_concepts = self.kb.intersect_from_iterables(cumulative_refinements[i],
                                                                             cumulative_refinements[j])
                    if ignore_union is False:
                        # 3.2 Union concepts having length i with concepts having length j.
                        union_of_concepts = self.kb.union_from_iterables(cumulative_refinements[i],
                                                                         cumulative_refinements[j])
                        res = union_of_concepts.union(intersect_of_concepts)
                    else:
                        res = intersect_of_concepts

                    # Store newly generated concepts at 3.2.
                    if len_ in cumulative_refinements:
                        x = cumulative_refinements[len_]
                        cumulative_refinements[len_] = x.union(res)
                    else:
                        if len_ in larger_cumulative_refinements:
                            x = larger_cumulative_refinements[len_]
                            larger_cumulative_refinements[len_] = x.union(res)
                        else:
                            larger_cumulative_refinements[len_] = res

        for k, v in larger_cumulative_refinements.items():
            yield from v

    def refine_top_concept(self) -> Iterable:
        # 1. Store all named classes including Nothing.
        generator_container = []
        all_subs = [i for i in self.kb.get_all_sub_concepts(self.kb.thing)]
        generator_container.append(all_subs)
        # 2. Negate (1) excluding Nothing and store it.
        generator_container.append(self.kb.negation_from_iterables((i for i in all_subs if i.name != '⊥')))

        # 3. Get all most general restrictions and store them \forall r. T, \exist r. T
        most_general_universal_restrictions = self.kb.most_general_universal_restrictions(self.kb.thing)
        most_general_existential_restrictions = self.kb.most_general_existential_restrictions(self.kb.thing)
        generator_container.append(most_general_universal_restrictions)
        generator_container.append(most_general_existential_restrictions)
        # 4. Generate all refinements of given concept that have length at ***** most max_length*****
        yield from self.apply_union_and_intersection_from_iterable(generator_container, ignore_union=False)

    def refine_atomic_concept(self, node: Node) -> Generator:
        yield self.kb.intersection(node.concept, self.kb.thing)

    def refine_complement_of(self, node: Node) -> Generator:
        parents = self.kb.get_direct_parents(self.kb.negation(node.concept))
        yield from self.kb.negation_from_iterables(parents)
        yield self.kb.intersection(node.concept, self.kb.thing)

    def refine_object_some_values_from(self, node: Node) -> Generator:
        assert isinstance(node.concept.filler, Concept)
        # rule 1: EXISTS r.D = > EXISTS r.E
        for i in self.refine(self.getNode(node.concept.filler, parent_node=node)):
            yield self.kb.existential_restriction(i, node.concept.role)

    def refine_object_all_values_from(self, node: Node):
        # rule 1: for all r.D = > for all r.E
        for i in self.refine(self.getNode(node.concept.filler, parent_node=node)):
            yield self.kb.universal_restriction(i, node.concept.role)

    def refine_object_union_of(self, node: Node):
        concept_A = node.concept.concept_a
        concept_B = node.concept.concept_b

        for ref_concept_A in self.refine(self.getNode(concept_A, parent_node=node)):
            yield self.kb.union(concept_B, ref_concept_A)

        for ref_concept_B in self.refine(self.getNode(concept_B, parent_node=node)):
            yield self.kb.union(concept_A, ref_concept_B)

    def refine_object_intersection_of(self, node: Node):
        concept_A = node.concept.concept_a
        concept_B = node.concept.concept_b
        for ref_concept_A in self.refine(self.getNode(concept_A, parent_node=node)):
            yield self.kb.intersection(concept_B, ref_concept_A)

        for ref_concept_B in self.refine(self.getNode(concept_B, parent_node=node)):
            yield self.kb.intersection(concept_A, ref_concept_B)

    def refine(self, node) -> Generator:
        assert isinstance(node, Node)
        if node.concept.is_atomic:
            if node.concept.name == '⊤':
                yield from self.top_refinements
            elif node.concept.name == '⊥':
                yield node.concept
            else:
                yield from self.refine_atomic_concept(node)
        elif node.concept.form == 'ObjectComplementOf':
            yield from self.refine_complement_of(node)
        elif node.concept.form == 'ObjectSomeValuesFrom':
            yield from self.refine_object_some_values_from(node)
        elif node.concept.form == 'ObjectAllValuesFrom':
            yield from self.refine_object_all_values_from(node)
        elif node.concept.form == 'ObjectUnionOf':
            yield from self.refine_object_union_of(node)
        elif node.concept.form == 'ObjectIntersectionOf':
            yield from self.refine_object_intersection_of(node)
        else:
            raise ValueError

    def getNode(self, c: Concept, parent_node=None, root=False):
        """

        @param c:
        @param parent_node:
        @param root:
        @return:
        """
        if parent_node is None and root is False:
            print(c)
            raise ValueError
        return Node(concept=c, parent_node=parent_node, root=root)

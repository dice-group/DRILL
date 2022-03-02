from typing import Iterable, Generator, Set
from owlapy.model import OWLClassExpression, OWLObjectComplementOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectUnionOf, OWLClass, OWLObjectIntersectionOf
from owlapy.render import DLSyntaxObjectRenderer


class RefinementOpt:
    def __init__(self, knowledge_base, n: int = 7):
        self.knowledge_base = knowledge_base
        self.max_length_of_top = n
        # sh↓ (A) = {A' \in N_C | A' \sqsubseteq A, there is no A'' \in N_C with A' \sqsubseteq A'' \sqsubseteq A}
        self.downward_subsumption_hierarchy = dict()
        self.all_individuals = frozenset(i for i in self.knowledge_base.individuals())
        self.m_top = self.refine_top(self.knowledge_base.thing)
        temp = set()
        for i in self.m_top:
            for j in self.m_top:
                temp.add(self.knowledge_base.union((i, j)))

        self.m_top.update(temp)
        del temp
        self.renderer = DLSyntaxObjectRenderer()

    def refine_top(self, class_expression) -> set:
        """ Refine Top Class Expression """
        # (1) {A | A \in N_C, A \sqcap B \not \equiv \bottom, A \sqcap B \not \equiv B, there is not A' \in N_C with A \sqsubset A' }
        # (2) {¬ A | A \in N_C, ¬A \sqcap B \not \equiv \bottom,  ¬A \sqcap B \not \equiv B, there is not A' \in N_C with A \sqsubset A' }
        # (3) {\exists r.T | mgr_B} ,mgr_B denotes the set of most general applicable roles w.r.t. B
        # (4) {\forall r.T | mgr_B} ,mgr_B denotes the set of most general applicable roles w.r.t. B

        # TODO: Room for parallelism
        one = refine_top_condition_one(class_expression, self.knowledge_base, self.all_individuals)
        two = refine_top_condition_two(class_expression, self.knowledge_base, self.all_individuals)
        three = refine_top_condition_three(class_expression, self.knowledge_base)
        four = refine_top_condition_four(class_expression, self.knowledge_base)

        refinements = one | two | three | four | {self.knowledge_base.nothing}
        return refinements

    def retrieve_top_refinement(self) -> Set[OWLClassExpression]:
        return self.m_top

    def refine_complement_of(self, class_expression: OWLObjectComplementOf) -> Set[OWLClassExpression]:
        assert isinstance(class_expression, OWLObjectComplementOf)
        # \rho_B(C) => if C =\notA s.t. A \in N_C
        # (1) ( \not A' | A' \in sh_down(A))
        # (2) \not A \sqcap D | D \in \rho_B (T)
        one = set(self.knowledge_base.negation_from_iterables(
            self.knowledge_base.get_direct_parents(class_expression.get_operand())))
        one.add(self.knowledge_base.intersection((class_expression, self.knowledge_base.thing)))
        return one

    def refine_object_intersection_of(self, class_expression: OWLObjectIntersectionOf) -> Set[OWLObjectIntersectionOf]:
        assert isinstance(class_expression, OWLObjectIntersectionOf)
        # \rho_B (C) => if C= C_1 \sqcap \dots C_n
        # (1) {C_1 \sqcap \dots C_{i-1} \sqcap D \sqcap C_{i+1} \dots C_n}
        one = set()
        child: OWLClassExpression
        operands: List[OWLClassExpression] = list(class_expression.operands())
        # Iterate over operands
        for i in range(len(operands)):
            concept_left, concept, concept_right = operands[:i], operands[i], operands[i + 1:]

            for ref_concept in self.refine(concept):
                if not ref_concept.is_owl_nothing():
                    intersection = self.knowledge_base.intersection(concept_left + [ref_concept] + concept_right)
                    one.add(intersection)

        return one

    def refine_object_union_of(self, class_expression: OWLObjectUnionOf) -> Set[OWLObjectIntersectionOf]:
        # \rho_B (C) => if C = C_1 \sqcup \dots C_n
        # (1) {C_1 \sqcup \dots C_{i-1} \sqcup D \sqcup C_{i+1} \dots C_n | D \in \rho_B (C_i), 1 \leq i \leq n}
        # (2) {(C_1 \sqcup \dots \sqcup C_N ) \sqcap D | D \in \rho_B (T)}

        assert isinstance(class_expression, OWLObjectUnionOf)

        child: OWLClassExpression
        operands: List[OWLClassExpression] = list(class_expression.operands())
        # Iterate over operands
        # (1)
        one = set()
        for i in range(len(operands)):
            concept_left, concept, concept_right = operands[:i], operands[i], operands[i + 1:]

            for ref_concept in self.refine(concept):
                # If concept is \bottom, there is no impact
                if not ref_concept.is_owl_nothing():
                    intersection = self.knowledge_base.union(concept_left + [ref_concept] + concept_right)
                    one.add(intersection)
        # (2)
        two = {self.knowledge_base.intersection((class_expression, self.knowledge_base.thing))}
        return one | two

    def refine_object_some_values_from(self, class_expression: OWLObjectSomeValuesFrom) -> Iterable[OWLClassExpression]:
        # \rho_B (C) => if C = \exist r.D
        # (1) {\exists r. E | A= ar(r), E \in p_A (D) }
        # (2) {\exists r. D \sqcap E ,  E \in p_B (T) }
        # (3) {\exists r. D | s \in sh_down(r)
        # (4) return (1) UNION (2) UNION (3)
        assert isinstance(class_expression, OWLObjectSomeValuesFrom)

        # (1) @TODO: Incomplete, get_object_property_ranges should not return an empty set
        one = set(i for i in self.knowledge_base.get_object_property_ranges(class_expression.get_property()))
        # (2)
        two = {self.knowledge_base.intersection((class_expression, self.knowledge_base.thing))}
        # (3))
        three = set()
        for i in self.refine(class_expression.get_filler()):
            three.add(self.knowledge_base.existential_restriction(i, class_expression.get_property()))
        return one | two | three

    def refine_object_all_values_from(self, class_expression: OWLObjectAllValuesFrom) -> Set[OWLObjectIntersectionOf]:
        """ """
        # \rho_B (C) => if C = \forall r.D
        # (1) {\forall r.E | A = ar(r), E \in \rho_A(D)}
        # (2) {\forall r.D \sqcap E | E \in \rho_B (\top)}
        # (3) {\forall r. \bottom | D = A \in N_C and sh_down (A) =\emptyset
        # (4) {\forall s. D | s \in sh_down(r) }

        assert isinstance(class_expression, OWLObjectAllValuesFrom)
        # (1) @TODO: Incomplete, get_object_property_ranges should not return an empty set
        one = set(i for i in self.knowledge_base.get_object_property_ranges(class_expression.get_property()))
        # (2)
        two = {self.knowledge_base.intersection((class_expression, self.knowledge_base.thing))}
        # (3) and (4) @TODO: Not available
        return one | two

    def refine(self, class_expression) -> Set[OWLClassExpression]:
        assert isinstance(class_expression, OWLClassExpression)

        if isinstance(class_expression, OWLObjectIntersectionOf):  # A \sqcap B s.t. A and B can be anything :)
            return self.refine_object_intersection_of(class_expression)
        elif isinstance(class_expression, OWLObjectUnionOf):  # A \sqcap B
            return self.refine_object_union_of(class_expression)
        elif class_expression.is_owl_thing():  # A= \top
            return self.retrieve_top_refinement()
        elif class_expression.is_owl_nothing():  # A= \bottom
            return {class_expression}
        elif isinstance(class_expression, OWLClass):  # A \in N_C (named concepts
            return self.refine_atomic_concept(class_expression)
        elif isinstance(class_expression, OWLObjectComplementOf):  # \neg A s.t. A \in N_C
            return self.refine_complement_of(class_expression)
        elif isinstance(class_expression, OWLObjectSomeValuesFrom):  # \exists r. C
            return self.refine_object_some_values_from(class_expression)
        elif isinstance(class_expression, OWLObjectAllValuesFrom):  # \forall r. C
            return self.refine_object_all_values_from(class_expression)
        else:
            raise KeyError(f'Could not understand its type: {str(class_expression)}')

    def refine_atomic_concept(self, class_expression: OWLClassExpression) -> Set[OWLClassExpression]:
        """ refine input name concept """
        # \rho_B (C) => if C = A s.t. A \in NC
        # (1) { A' | A' \in sh_down (A)}
        # (2) { A \sqcap D | D \in P_B(T)}
        one = refine_top_condition_one(class_expression, self.knowledge_base,
                                       frozenset(i for i in self.knowledge_base.individuals(class_expression)))
        two = set(self.knowledge_base.intersection((i, class_expression)) for i in self.retrieve_top_refinement())
        return one | two


def refine_top_condition_one(class_expression, knowledge_base, all_individuals) -> set:
    # (1.1) {A | A \in N_C ,s.t. \neg \exist A' \in N_C A \sqsubset A'}.
    one = set()
    for i in knowledge_base.get_direct_sub_concepts(class_expression):
        individuals = frozenset(i for i in knowledge_base.individuals(i))
        # (1.2) A \sqcap \top \not \equiv \bottom.
        if len(individuals & all_individuals) > 0:
            # (1.3) The condition (A \sqcap \top \not \equiv T) is ignored.
            # because I do not understand it
            # From my understanding (PERSON \sqcap \top \equiv \top) in the family dataset
            # Yet, DL-learner rho(T) returns PERSON.
            one.add(i)
    return one


def refine_top_condition_two(class_expression, knowledge_base, all_individuals):
    two = set()
    # (2) {¬ A | A \in N_C, there is not A' \in N_C with A \sqsubset A' }, i.e., negated leaf nodes.
    for i in knowledge_base.negation_from_iterables(knowledge_base.get_leaf_concepts(class_expression)):
        individuals = frozenset(i for i in knowledge_base.individuals(i))
        # (2.1) \negA \sqcap \top \not \equiv \bottom.
        if len(individuals & all_individuals) > 0:
            # (2.2) \negA \sqcap \top \not \equiv \top.
            if individuals != all_individuals:
                two.add(i)
    return two


def refine_top_condition_three(class_expression, knowledge_base):
    # (3) {\exists r.T | mgr_B} ,mgr_B denotes the set of most general applicable roles w.r.t. B
    return frozenset(knowledge_base.most_general_existential_restrictions(domain=class_expression))


def refine_top_condition_four(class_expression, knowledge_base):
    # (3) {\forall r.T | mgr_B} ,mgr_B denotes the set of most general applicable roles w.r.t. B
    return frozenset(knowledge_base.most_general_universal_restrictions(domain=class_expression))

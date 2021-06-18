from owlready2 import Ontology, World, Thing
import owlready2
from .concept_generator import ConceptGenerator
from .concept import Concept
from typing import Generator, Iterable
from .util import parametrized_performance_debugger
from .abstracts import AbstractKnowledgeBase
from .data_struct import PropertyHierarchy
import warnings
from .static_funcs import parse_tbox_into_concepts

warnings.filterwarnings("ignore")
import multiprocessing


class KnowledgeBase(AbstractKnowledgeBase):
    """ Knowledge Base Class representing Tbox and Abox along with the concept hierarchy """

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.world = World()
        self.onto = self.world.get_ontology('file://' + self.path).load(reload=True)
        self.property_hierarchy = PropertyHierarchy(self.onto)
        self.name = self.onto.name
        self.parse()
        self._concept_generator = ConceptGenerator(concepts=self.uri_to_concept,
                                                   thing=self.thing,
                                                   nothing=self.nothing,
                                                   onto=self.onto)

        self.describe()

    def instance_retrieval(self, c: Concept):
        if c.instances is None:
            return self._concept_generator.instance_retrieval(c)
        return c.instances

    def instance_retrieval_from_iterable(self, nodes: Iterable):
        return [self.instance_retrieval(n.concept) for n in nodes]

    def instance_retrieval_parallel_from_iterable(self, nodes: Iterable):
        """
        with multiprocessing.Pool(processes=4) as executor:
            instances = executor.map(self.concept_generator.instance_retrieval_node, nodes)
        return instances

        with concurrent.futures.ThreadPoolExecutor() as executor:
            instances = executor.map(self.concept_generator.instance_retrieval_node, nodes)
        return instances

        => The least efficient.
        with concurrent.futures.ProcessPoolExecutor() as executor:
            instances = executor.map(self.concept_generator.instance_retrieval_node, nodes)
        return instances
        """
        with multiprocessing.Pool(processes=4) as executor:
            instances = executor.map(self._concept_generator.instance_retrieval_node, nodes)
        return instances

    def clean(self):
        """
        Clearn all stored values if there is any.
        @return:
        """

    def __concept_hierarchy_fill(self, owl_concept, our_concept):
        """
        our_concept can not be Nothing or Thing
        """
        has_sub_concept = False

        # 3. Get all sub concepts of input concept.
        for owlready_subclass_concept_A in owl_concept.descendants(include_self=False):
            if owlready_subclass_concept_A.name in ['Nothing', 'Thing', 'T', '⊥']:
                raise ValueError
            has_sub_concept = True
            # 3.2 Map them into the corresponding our Concept objects.
            subclass_concept_A = self.uri_to_concept[owlready_subclass_concept_A.iri]
            # 3.3. Add all our sub concepts into the concept the top down concept hierarchy.
            self.top_down_concept_hierarchy[our_concept].add(subclass_concept_A)
            self.down_top_concept_hierarchy[subclass_concept_A].add(our_concept)

        # 4. Get all super concepts of input concept.
        for owlready_superclass_concept_A in owl_concept.ancestors(include_self=False):
            if owlready_superclass_concept_A.name == 'Thing' and len(
                    [i for i in owl_concept.ancestors(include_self=False)]) == 1:
                self.top_down_direct_concept_hierarchy[self.thing].add(our_concept)
                self.down_top_direct_concept_hierarchy[our_concept].add(self.thing)
            else:
                # 3.2 Map them into the corresponding our Concept objects.
                superclass_concept_A = self.uri_to_concept[owlready_superclass_concept_A.iri]
                # 3.3. Add all our super concepts into the concept the down to concept concept hierarchy.
                self.down_top_concept_hierarchy[our_concept].add(superclass_concept_A)
                self.top_down_concept_hierarchy[superclass_concept_A].add(our_concept)

        # 4. If concept does not have any sub concept, then concept is a leaf concept.
        #  Every leaf concept is directly related to Nothing.
        if has_sub_concept is False:
            self.top_down_direct_concept_hierarchy[our_concept].add(self.nothing)
            self.down_top_direct_concept_hierarchy[self.nothing].add(our_concept)

    def __direct_concept_hierarchy_fill(self, owlready_concept_A, concept_A, onto):
        for owlready_direct_subclass_concept_A in owlready_concept_A.subclasses(
                world=onto.world):  # returns direct subclasses
            if owlready_concept_A == owlready_direct_subclass_concept_A:
                print(owlready_concept_A)
                print(owlready_direct_subclass_concept_A)
                raise ValueError
            direct_subclass_concept_A = self.uri_to_concept[owlready_direct_subclass_concept_A.iri]

            self.top_down_direct_concept_hierarchy[concept_A].add(direct_subclass_concept_A)
            self.down_top_direct_concept_hierarchy[direct_subclass_concept_A].add(concept_A)

    def __build_hierarchy(self, onto: Ontology) -> None:
        """
        Builds concept sub and super classes hierarchies.

        1) self.top_down_concept_hierarchy is a mapping from Concept objects to a set of Concept objects that are
        direct subclasses of given Concept object.

        2) self.down_top_concept_hierarchy is a mapping from Concept objects to set of Concept objects that are
        direct superclasses of given Concept object.
        """
        # 1. (Mapping from string URI to Class Expressions, Thing Concept, Nothing Concept
        self.uri_to_concept, self.thing, self.nothing = parse_tbox_into_concepts(onto)
        assert len(self.uri_to_concept) > 2

        assert self.thing.iri == 'http://www.w3.org/2002/07/owl#Thing'
        assert self.thing.name == '⊤'

        assert self.nothing.iri == 'http://www.w3.org/2002/07/owl#Nothing'
        assert self.nothing.name == '⊥'

        self.individuals = self.thing.instances
        self.down_top_concept_hierarchy[self.thing] = set()

        for IRI, concept_A in self.uri_to_concept.items():  # second loop over concepts in the execution,
            assert IRI == concept_A.iri
            try:
                assert len(onto.search(iri=IRI)) == 1
            except AssertionError:
                # Thing and Nothing is not added into hierarchy
                assert IRI in ['http://www.w3.org/2002/07/owl#Thing','http://www.w3.org/2002/07/owl#Nothing']
                assert concept_A.name in ['⊤', '⊥']
                continue
            owlready_concept_A = onto.search(iri=concept_A.iri)[0]
            assert owlready_concept_A.iri == concept_A.iri
            self.__concept_hierarchy_fill(owlready_concept_A, concept_A)
            self.__direct_concept_hierarchy_fill(owlready_concept_A, concept_A, onto)

            # All concepts are subsumed by Thing.
            self.top_down_concept_hierarchy[self.thing].add(concept_A)
            self.down_top_concept_hierarchy[concept_A].add(self.thing)

            # All concepts subsume Nothing.
            self.top_down_concept_hierarchy[concept_A].add(self.nothing)
            self.down_top_concept_hierarchy[self.nothing].add(concept_A)

        self.top_down_concept_hierarchy[self.thing].add(self.nothing)
        self.down_top_concept_hierarchy[self.nothing].add(self.thing)

        ################################################################################################################
        # Sanity checking
        # 1. Did we parse classes correctly ?
        owlready2_classes = {i.iri for i in onto.classes()}
        our_classes = {k for k, v in self.uri_to_concept.items()}
        try:
            assert our_classes.issuperset(owlready2_classes) and (
                    our_classes.difference(owlready2_classes) == {'http://www.w3.org/2002/07/owl#Thing',
                                                                  'http://www.w3.org/2002/07/owl#Nothing'})
        except AssertionError:
            raise AssertionError('Assertion error => at superset checking.')

        try:
            # Thing subsumes all parsed concept except itself.
            assert len(self.top_down_concept_hierarchy[self.thing]) == (len(our_classes) - 1)
            assert len(self.down_top_concept_hierarchy[self.nothing]) == (len(our_classes) - 1)
        except AssertionError:
            raise AssertionError('Assertion error => at concept hierarchy checking.')

        # start from here
        try:
            assert len(self.down_top_concept_hierarchy[self.nothing]) == (len(our_classes) - 1)
            assert len(self.top_down_direct_concept_hierarchy[self.thing]) >= 1
        except AssertionError:
            raise AssertionError('Assertion error => total number of parsed concept checking')

        # 2. Did we create top down direct concept hierarchy correctly ?
        for concept, direct_sub_concepts in self.top_down_direct_concept_hierarchy.items():
            for dsc in direct_sub_concepts:
                assert concept.instances.issuperset(dsc.instances)

        # 3. Did we create top down concept hierarchy correctly ?
        for concept, direct_sub_concepts in self.top_down_concept_hierarchy.items():
            for dsc in direct_sub_concepts:
                assert concept.instances.issuperset(dsc.instances)

        # 3. Did we create down top direct concept hierarchy correctly ?
        for concept, direct_super_concepts in self.down_top_direct_concept_hierarchy.items():
            for dsc in direct_super_concepts:
                assert concept.instances.issubset(dsc.instances)

        # 4. Did we create down top concept hierarchy correctly ?
        for concept, direct_super_concepts in self.down_top_concept_hierarchy.items():
            for dsc in direct_super_concepts:
                try:
                    assert concept.instances.issubset(dsc.instances)
                except AssertionError:
                    raise AssertionError('Subset error')

    def parse(self):
        """
        Top-down and bottom up hierarchies are constructed from from owlready2.Ontology
        """
        self.__build_hierarchy(self.onto)

    # OPERATIONS
    def negation(self, concept: Concept) -> Concept:
        """ Return a Concept object that is a negation of given concept."""
        return self._concept_generator.negation(concept)

    def union(self, conceptA: Concept, conceptB: Concept) -> Concept:
        """Return a concept c == (conceptA OR conceptA)"""
        return self._concept_generator.union(conceptA, conceptB)

    def intersection(self, conceptA: Concept, conceptB: Concept) -> Concept:
        """Return a concept c == (conceptA AND conceptA)"""
        return self._concept_generator.intersection(conceptA, conceptB)

    def existential_restriction(self, concept: Concept, property_) -> Concept:
        """Return a concept c == (\exists R.C)"""
        return self._concept_generator.existential_restriction(concept, property_)

    def universal_restriction(self, concept: Concept, property_) -> Concept:
        """Return a concept c == (\forall R.C)"""
        return self._concept_generator.universal_restriction(concept, property_)


    @staticmethod
    def is_atomic(c: owlready2.entity.ThingClass):
        """
        Check whether input owlready2 concept object is atomic concept.
        This is a workaround
        @param c:
        @return:
        """
        assert isinstance(c, owlready2.entity.ThingClass)
        if '¬' in c.name and not (' ' in c.name):
            return False
        elif ' ' in c.name or '∃' in c.name or '∀' in c.name:
            return False
        else:
            return True

    def get_leaf_concepts(self, concept: Concept) -> Generator:
        """ Return : { x | (x subClassOf concept) AND not exist y: y subClassOf x )} """
        assert isinstance(concept, Concept)
        for leaf in self.concepts_to_leafs[concept]:
            yield leaf

    @parametrized_performance_debugger()
    def negation_from_iterables(self, s: Iterable) -> Generator:
        """ Return : { x | ( x \equiv not s} """
        assert isinstance(s, Iterable)
        for item in s:
            yield self._concept_generator.negation(item)

    # @parametrized_performance_debugger()
    def get_direct_sub_concepts(self, concept: Concept) -> Generator:
        """ Return : { x | ( x subClassOf concept )} """
        assert isinstance(concept, Concept)
        yield from self.top_down_direct_concept_hierarchy[concept]

    def get_all_sub_concepts(self, concept: Concept):
        """ Return : { x | ( x subClassOf concept ) OR ..."""
        assert isinstance(concept, Concept)
        yield from self.top_down_concept_hierarchy[concept]

    def get_direct_parents(self, concept: Concept) -> Generator:
        """ Return : { x | (concept subClassOf x)} """
        assert isinstance(concept, Concept)
        yield from self.down_top_direct_concept_hierarchy[concept]

    def get_parents(self, concept: Concept) -> Generator:
        """ Return : { x | (concept subClassOf x)} """
        yield from self.down_top_concept_hierarchy[concept]

    def most_general_existential_restrictions(self, concept: Concept) -> Generator:
        """ Return : { \exist.r.x | r \in MostGeneral r} """
        assert isinstance(concept, Concept)
        for prob in self.property_hierarchy.get_most_general_property():
            yield self._concept_generator.existential_restriction(concept, prob)

    def union_from_iterables(self, concept_a: Iterable, concept_b: Iterable):
        temp = set()
        seen = set()
        for i in concept_a:
            for j in concept_b:
                if (i.name, j.name) in seen:
                    continue

                u = self._concept_generator.union(i, j)
                seen.add((i.name, j.name))
                seen.add((j.name, i.name))
                temp.add(u)
        return temp

    def intersect_from_iterables(self, concept_a, concept_b):
        temp = set()
        seen = set()
        for i in concept_a:
            for j in concept_b:
                if (i.name, j.name) in seen:
                    continue

                and_ = self._concept_generator.intersection(i, j)
                seen.add((i.name, j.name))
                seen.add((j.name, i.name))
                temp.add(and_)
        return temp

    def most_general_universal_restrictions(self, concept: Concept) -> Generator:
        """ Return : { \forall.r.x | r \in MostGeneral r} """
        assert isinstance(concept, Concept)
        for prob in self.property_hierarchy.get_most_general_property():
            yield self._concept_generator.universal_restriction(concept, prob)

# from util import jaccard_similarity, compute_prediction, evaluate_results
import concurrent.futures

import dicee
import requests

from abc import ABC, abstractmethod
from typing import Set, Iterable

from abc import ABC
from owlapy.parser import DLSyntaxParser
from owlapy.owl2sparql.converter import Owl2SparqlConverter
import time
import asyncio


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


class AbstractReasoner(ABC):
    def predict(self, concept) -> Set[str]:
        if isinstance(concept, NC):
            return self.atomic_concept(concept)
        elif isinstance(concept, NNC):
            return self.negated_atomic_concept(concept)
        elif isinstance(concept, ConjunctionDLConcept):
            return self.conjunction(concept)
        elif isinstance(concept, DisjunctionDLConcept):
            return self.disjunction(concept)
        elif isinstance(concept, Restriction):
            return self.restriction(concept)
        elif isinstance(concept, ValueRestriction):
            return self.value_restriction(concept)
        else:
            raise NotImplementedError(type(concept))

    @abstractmethod
    def atomic_concept(self, concept: NC) -> Set[str]:
        raise NotImplementedError()

    @abstractmethod
    def negated_atomic_concept(self, concept: NNC) -> Set[str]:
        raise NotImplementedError()

    @abstractmethod
    def restriction(self, concept: Restriction) -> Set[str]:
        raise NotImplementedError()

    @abstractmethod
    def conjunction(self, concept: NC) -> Set[str]:
        raise NotImplementedError()

    @abstractmethod
    def disjunction(self, concept: NC) -> Set[str]:
        raise NotImplementedError()


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


from owlapy.model import OWLThing, OWLNothing, OWLClass, IRI, OWLObjectUnionOf, OWLObjectIntersectionOf


class SPARQLCWR(AbstractReasoner):
    def __init__(self, url, name: str = 'sparqlcwr'):
        super(SPARQLCWR, self)
        self.url = url
        self.name = name
        self.thing = OWLThing
        self.nothing = OWLNothing
        # (1) Find all named concepts:Assume that the forward-chain is applied
        self.iri_named_concepts = self.query("PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
                                             "SELECT DISTINCT ?var\n"
                                             "WHERE {?var a owl:Class.}")
        # (3) Detect the name space.
        self.namespace = None
        # How do you get base IRI ?
        for i in self.iri_named_concepts:
            if self.namespace is None:
                # start 1 because of the left bracket <...>
                self.namespace = i[1:i.index('#') + 1]
            else:
                assert self.namespace == i[1:i.index('#') + 1]
        self.parser = DLSyntaxParser(self.namespace)
        self.converter = Owl2SparqlConverter()
        self.named_concepts = set()
        for i in self.iri_named_concepts:
            remainder = i[i.index(self.namespace) + len(self.namespace):-1]
            self.named_concepts.add(OWLClass(iri=IRI(namespace=self.namespace, remainder=remainder)))

        # (2) Find all named individuals.
        self.str_individuals = self.query("PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
                                          "SELECT DISTINCT ?var\n"
                                          "WHERE {?var a owl:NamedIndividual.}")
        if self.str_individuals == self.query("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                                              "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
                                              "SELECT DISTINCT ?var\n"
                                              "WHERE {\n"
                                              "?var a ?type.\n"
                                              "?type rdfs:subClassOf ?y.\n"
                                              "FILTER NOT EXISTS {?type rdf:type rdf:Class}\n"
                                              "FILTER NOT EXISTS {?type rdf:type rdf:Property}}"):
            self.materialized = True
        else:
            self.materialized = False

    def __str__(self):
        return f"SPARQLCWR:|Individuals|:{len(self.individuals)}\t |Concepts|:{len(self.concepts)}"

    def sanity_checking(self):
        # https://stackoverflow.com/questions/22190403/how-could-i-use-requests-in-asyncio
        for i in self.named_concepts:
            for j in self.named_concepts:
                if i == j:
                    continue
                print(i, j)
                x = OWLObjectUnionOf(operands=[i, i])
                print(f"{x}\t{len(self.retrieve(x))}")
                x = OWLObjectUnionOf(operands=[i, j])
                print(f"{x}\t{len(self.retrieve(x))}")
                x = OWLObjectUnionOf(operands=[self.thing, i])
                print(f"{x}\t{len(self.retrieve(x))}")
                x = OWLObjectUnionOf(operands=[i, self.nothing])
                print(f"{x}\t{len(self.retrieve(x))}")
                x = OWLObjectUnionOf(operands=[self.thing, self.nothing])
                print(f"{x}\t{len(self.retrieve(x))}")

    def sparql(self, cls: OWLClass):
        return self.converter.as_query("?var", cls, False)

    @property
    def individuals(self):
        return self.str_individuals

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
            print('Errr at SPARQL conversion')
            return set()
        return self.query(sparql_mapping)

    def apply_construction_rules(self, concept):
        results = set()
        for i in self.named_concepts:
            results.add(OWLObjectUnionOf(operands=[i, concept]))
            results.add(OWLObjectIntersectionOf(operands=[i, concept]))
        return results

    def old_retrieve(self, concept) -> Set[str]:
        """
        perform concept retrieval
        :param concept:
        :return:
        """
        if isinstance(concept, Restriction) and concept.opt == '∀':
            # A concept retrieval for ∀ r.C is performed by \neg \∃ r. ∃C
            # given '∀' r.C, convert it into \neg r. \neg C
            sparql_query = converter.as_query("?var", parser.parse_expression(
                Restriction(opt="∃", role=concept.role_iri, filler=concept.filler.neg()).str), False)
            return self.all_individuals.difference(self.query(sparql_query))
        else:
            sparql_query = converter.as_query("?var", parser.parse_expression(concept.str), False)
            return self.query(sparql_query)

    def atomic_concept(self, concept: NC) -> Set[str]:
        """ {x | f(x,type,concept) \ge \gamma} """
        assert isinstance(concept, NC)
        return self.retrieve(concept)

    def negated_atomic_concept(self, concept: NNC) -> Set[str]:
        assert isinstance(concept, NNC)
        return self.retrieve(concept)

    def conjunction(self, concept) -> Set[str]:
        """  Conjunction   (⊓) : C ⊓ D  : C^I ⊓ D^I """
        return self.retrieve(concept)

    def disjunction(self, concept) -> Set[str]:
        """  Disjunction   (⊔) : C ⊔ D  : C^I ⊔ D^I """
        return self.retrieve(concept)

    def restriction(self, concept):
        return self.retrieve(concept)

    def value_restriction(self, concept: ValueRestriction) -> Set[str]:
        return self.retrieve(concept)


class CWR(AbstractReasoner):
    """ Closed World Assumption"""

    def __init__(self, database, all_named_individuals):
        super(CWR, self)
        self.database = database
        self.all_named_individuals = all_named_individuals
        self.name = 'CWR'

    def atomic_concept(self, concept: NC) -> Set[str]:
        """ Atomic concept (C)    C^I \⊆ ∆^I """
        assert isinstance(concept, NC)
        return set(self.database[(self.database['relation'] == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>') & (
                self.database['object'] == concept.iri)]['subject'].tolist())

    def negated_atomic_concept(self, concept: NNC) -> Set[str]:
        """ Negation (¬C) : ∆^I \ C^I  """
        assert isinstance(concept, NNC)
        return self.all_named_individuals - self.atomic_concept(concept=concept.neg())

    def conjunction(self, concept: ConjunctionDLConcept) -> Set[str]:
        """  Conjunction   (⊓) : C ⊓ D  : C^I ⊓ D^I """
        assert isinstance(concept, ConjunctionDLConcept)
        return self.atomic_concept(concept.left).intersection(self.atomic_concept(concept.right))

    def disjunction(self, concept: DisjunctionDLConcept) -> Set[str]:
        """  Disjunction   (⊔) : C ⊔ D  : C^I ⊔ D^I """
        assert isinstance(concept, DisjunctionDLConcept)
        return self.atomic_concept(concept.left).union(self.atomic_concept(concept.right))

    def restriction(self, concept: Restriction) -> Set[str]:
        if concept.opt == '∃':
            return self.existential_restriction(role=concept.role_iri, filler_concept=concept.filler)
        elif concept.opt == '∀':
            return self.universal_restriction(role=concept.role_iri, filler_concept=concept.filler)
        else:
            raise ValueError(concept.str)

    def existential_restriction(self, role: str, filler_concept, filler_indv: Set[str] = None):
        """ \exists r.C  { x \mid \exists y. (x,y) \in r^I \land y \in C^I } """
        # (1) All triples with a given role.
        triples_with_role = self.database[self.database['relation'] == role].to_records(index=False)
        # (2) All individuals having type C.
        if filler_concept:
            filler_individuals = self.predict(concept=filler_concept)
        elif filler_indv:
            filler_individuals = filler_indv
        else:
            filler_individuals = self.all_named_individuals
        # (3) {x | ( x r y ) and (y type C) }.
        return {spo[0] for spo in triples_with_role if spo[2] in filler_individuals}

    def universal_restriction(self, role: str, filler_concept: AbstractDLConcept):
        """ \forall r.C   &
        { x | \forall y. (x,y) \in r^I \implies y \in C^I \}  """
        # READ Towards SPARQL - Based Induction for Large - Scale RDF Data sets Technical Report for the details
        # http://svn.aksw.org/papers/2016/ECAI_SPARQL_Learner/tr_public.pdf
        raise NotImplementedError('Rewrite this part by using existential')
        results = set()
        filler_individuals = self.predict(filler_concept)

        domain_of_interpretation_of_relation = {_ for _ in self.database[(self.database['relation'] == role)][
            'subject'].tolist()}

        for i in domain_of_interpretation_of_relation:
            # {SELECT ?var
            #   (count(?s2) as ?cnt2)
            #   WHERE {?var r ?s2}
            #   GROUP By ?var}
            # All objects given subject and a relation
            cnt2 = {_ for _ in self.database[(self.database['subject'] == i) & (self.database['relation'] == role)][
                'object'].tolist()}
            # {SELECT ?var
            #   (count (?s1) as ?cn1)
            #   WHERE { ?var r ?s1 .
            #           \tau(C,?s1) .}
            #   GROUP BY ?var }
            cnt1 = cnt2.intersection(filler_individuals)
            if len(cnt1) == len(cnt2):  # or cnt1==cnt2
                results.add(i)
        return results


class NWR(AbstractReasoner):
    def __init__(self, predictor: dicee.KGE, gammas=None, all_named_individuals: Set[str] = None):
        super(NWR, self)
        self.predictor = predictor
        self.gammas = gammas
        self.all_named_individuals = all_named_individuals
        self.name = 'nwr'

    def atomic_concept(self, concept: NC) -> Set[str]:
        """ {x | f(x,type,concept) \ge \gamma} """
        assert isinstance(concept, NC)
        # (1) Compute scores for all entities.
        scores_for_all = self.predictor.predict(relations=['<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'],
                                                tail_entities=[concept.iri])
        # (2) Iterative (1) and return entities whose predicted score satisfies the condition.
        raw_results = {self.predictor.idx_to_entity[index] for index, flag in
                       enumerate(scores_for_all >= self.gammas['NC']) if
                       flag}
        # (3) Remove non entity predictions.
        return {i for i in raw_results if i in self.all_named_individuals}

    def negated_atomic_concept(self, concept: NNC) -> Set[str]:
        assert isinstance(concept, NNC)
        return self.all_named_individuals - self.predict(concept.neg())

    def conjunction(self, concept: ConjunctionDLConcept) -> Set[str]:
        """ 	Conjunction   ⊓           & $C\sqcap D$    & $C^\mathcal{I}\cap D^\mathcal{I} """
        return self.predict(concept=concept.left).intersection(self.predict(concept=concept.right))

    def disjunction(self, concept: DisjunctionDLConcept) -> Set[str]:
        """ ⊔ """
        return self.predict(concept=concept.left).union(self.predict(concept=concept.right))

    def restriction(self, concept: Restriction) -> Set[str]:
        if concept.opt == '∃':
            return self.existential_restriction(role=concept.role_iri, filler_concept=concept.filler)
        elif concept.opt == '∀':
            return self.universal_restriction(role=concept.role_iri, filler_concept=concept.filler)
        else:
            raise ValueError(concept.str)

    def existential_restriction(self, role: str, filler_concept: str = None, filler_individuals: Set[str] = None):
        """ \exists r.C  { x \mid \exists y. (x,y) \in r^I \land y \in C^I }

        ∃ hasSibling.Female
                {?var r ?s.} ∪ {?s type C}

        SELECT DISTINCT ?var
        WHERE {
                    ?var <http://www.benchmark.org/family#hasSibling> ?s_1 .
                    (1) ?s_1 a <http://www.benchmark.org/family#Female> .  }


        """
        # (1) Find individuals that are likeliy y \in C^I.
        if filler_concept:
            filler_individuals = self.predict(concept=filler_concept)
        elif filler_individuals is not None:
            filler_individuals = filler_individuals
        else:
            filler_individuals = self.all_named_individuals

        results = set()
        # (2) For each filler individual
        for i in filler_individuals:
            # (2.1) Assign scores for all subjects.
            scores_for_all = self.predictor.predict(relations=[role], tail_entities=[i])
            ids = (scores_for_all >= self.gammas['Exists']).nonzero(as_tuple=True)[0].tolist()
            if len(ids) >= 1:
                results.update(set(ids))
            else:
                continue
        return {self.predictor.idx_to_entity[i] for i in results}

    def old_universal_restriction(self, role: str, filler_concept: AbstractDLConcept):

        results = set()
        interpretation_of_filler = self.predict(filler_concept)
        # We should only consider the domain of the interpretation of the role.
        for i in self.all_named_individuals:
            # {SELECT ?var
            #   (count(?s2) as ?cnt2)
            #   WHERE { ?var r ?s2 }
            #   GROUP By ?var}
            scores_for_all = self.predictor.predict(head_entities=[i], relations=[role]).flatten()
            raw_results = {self.predictor.idx_to_entity[index] for index, flag in
                           enumerate(scores_for_all >= self.gammas['Forall']) if flag}
            # Demir hasSibling {......}
            cnt2 = {i for i in raw_results if i in self.all_named_individuals}

            # {SELECT ?var
            #   (count (?s1) as ?cn1)
            #   WHERE { ?var r ?s1 .
            #           \tau(C,?s1) .}
            #   GROUP BY ?var }
            # Demir hasSibling {......}
            cnt1 = cnt2.intersection(interpretation_of_filler)

            cnt1_and_cnt2 = cnt1.intersection(cnt2)
            cnt1_or_cnt2 = cnt1.union(cnt2)
            if len(cnt1_and_cnt2) == 0 and len(cnt1_or_cnt2) == 0:
                # if both empty
                results.add(i)
            elif len(cnt1_and_cnt2) == 0 or len(cnt1_or_cnt2) == 0:
                # if only one of them is empty
                continue
            elif len(cnt1_and_cnt2) / len(cnt1_or_cnt2) >= self.gammas['Forall']:
                # if none of them empty
                results.add(i)
            else:
                continue
        return results

    def universal_restriction(self, role: str, filler_concept: AbstractDLConcept):
        return self.all_named_individuals.difference(
            self.existential_restriction(role=role, filler_concept=filler_concept.neg()))

    def value_restriction(self, concept: ValueRestriction):
        results = dict()
        for i in self.predict(concept.filler):
            scores_for_all = self.predictor.predict(relations=[concept.role_iri], tail_entities=[i])

            # (2) Iterative (1) and return entities whose predicted score satisfies the condition.
            raw_results = {self.predictor.idx_to_entity[index] for index, flag in
                           enumerate(scores_for_all >= self.gammas['Value']) if flag}
            # (3) Remove non entity predictions.
            for k in raw_results:
                if k in self.all_named_individuals:
                    results.setdefault(k, 1)
                    results[k] += 1
        if concept.opt == '≥':
            # at least
            return {k for k, v in results.items() if v >= concept.val}
        else:
            return {k for k, v in results.items() if v <= concept.val}

    def find_gammas(self, gammas, concepts, true_func):
        for (name, i) in concepts:
            print(f'Searching gamma for {name}')
            # Finding a good gamma
            best_sim = 0.0
            best_gamma = 0.0
            for gamma in gammas:
                self.gammas[name] = gamma
                df = evaluate_results(true_results=compute_prediction(i, predictor=true_func),
                                      predictions=compute_prediction(i, predictor=self))

                avg_sim = df[['Similarity']].mean().values[0]
                # print(f"Gamma:{gamma}\t Sim:{avg_sim}")
                if avg_sim > best_sim:
                    best_gamma = gamma
                    best_sim = avg_sim
                    print(f"Current Best Gamma:{best_gamma}\t for {name} Sim:{best_sim}")

            print(f"Best Gamma:{best_gamma}\t for {name} Sim:{best_sim}")
            self.gammas[name] = best_gamma


class HermiT(AbstractReasoner):
    def __init__(self, url):
        super(HermiT, self)
        self.url = url
        self.name = 'HermiT'

        # TODO:
        # return all concepts
        # return most general concepts

    def retrieve(self, concept) -> Set[str]:
        """
        perform concept retrieval
        :param concept:
        :return:
        """
        try:
            return {i for i in
                    requests.post('http://localhost:8080/hermit', data=concept.manchester_str).json()['individuals']}
        except requests.exceptions.JSONDecodeError:
            print('JSON Decoding Error')
            print(concept.manchester_str)
            return set()

    def atomic_concept(self, concept: NC) -> Set[str]:
        """ {x | f(x,type,concept) \ge \gamma} """
        assert isinstance(concept, NC)
        return self.retrieve(concept)

    def negated_atomic_concept(self, concept: NNC) -> Set[str]:
        assert isinstance(concept, NNC)
        return self.retrieve(concept)

    def conjunction(self, concept) -> Set[str]:
        """  Conjunction   (⊓) : C ⊓ D  : C^I ⊓ D^I """
        return self.retrieve(concept)

    def disjunction(self, concept) -> Set[str]:
        """  Disjunction   (⊔) : C ⊔ D  : C^I ⊔ D^I """
        return self.retrieve(concept)

    def restriction(self, concept):
        return self.retrieve(concept)

    def value_restriction(self, concept: ValueRestriction) -> Set[str]:
        return self.retrieve(concept)

from .abstracts import BaseConcept

class Concept(BaseConcept):
    """
    Concept Class representing Concepts in Description Logic, Classes in OWL.
    """

    def __init__(self, *, name, instances, form, iri=None, concept_a=None, concept_b=None, filler=None, role=None):
        super().__init__(name=name, iri=iri, form=form, instances=instances, concept_a=concept_a, concept_b=concept_b,
                         role=role, filler=filler)

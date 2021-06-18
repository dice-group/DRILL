from typing import Iterable, Tuple, Dict,Any
from collections import deque
from owlready2 import Thing, Nothing
def concepts_sorter(A, B):
    if len(A) < len(B):
        return A, B
    if len(A) > len(B):
        return B, A

    args = [A, B]
    args.sort(key=lambda ce: ce.name)
    return args[0], args[1]


def retrieve_concept_chain(node):
    """
    Given a node return its parent hierarchy
    @param node:
    @return:
    """
    hierarchy = deque()
    if node.parent_node:
        hierarchy.appendleft(node.parent_node)
        while hierarchy[-1].parent_node is not None:
            hierarchy.append(hierarchy[-1].parent_node)
        hierarchy.appendleft(node)
    return hierarchy

def parse_tbox_into_concepts(onto) -> Tuple[Dict, Any, Any]:
    """
    Workaround Any=> Concept.
    """
    from .concept import Concept
    iri_to_concept = dict()
    individuals = set()
    #⊤ ⊥
    thing = Concept(name='⊤', form='Class', iri=str(Thing.iri),
                    instances={i.iri for i in Thing.instances(world=onto.world)})
    nothing = Concept(name='⊥', form='Class', iri=str(Nothing.iri), instances=set())
    # if "Owlready2 * Warning: ignoring cyclic subclass of/subproperty of, involving:" occurs
    # onto.classes() neither starts nor ends.
    print('Iterate over classes')
    for c in onto.classes():
        temp_concept = Concept(name=c.name, iri=c.iri, form='Class',
                               instances={i.iri for i in c.instances(world=onto.world)})
        iri_to_concept[temp_concept.iri] = temp_concept
        individuals.update(temp_concept.instances)
    try:
        assert thing.instances  # if empty throw error.
        assert individuals.issubset(thing.instances)
    except AssertionError:
        # print('Sanity checking failed: owlready2.Thing does not contain any individual. To alleviate this issue, we explicitly assign all individuals/instances to concept T.')
        thing.instances = individuals

    iri_to_concept[thing.iri] = thing
    iri_to_concept[nothing.iri] = nothing
    return iri_to_concept, thing, nothing

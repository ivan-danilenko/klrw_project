from typing import Callable, Iterable, Any

from sage.combinat.root_system.cartan_type import CartanType_abstract

from klrw.framed_dynkin import (
    FramedDynkinDiagram_class,
    FramedDynkinDiagram_with_dimensions,
    NodeInFramedQuiver,
)
from klrw.klrw_algebra import KLRWAlgebra
from klrw.klrw_state import KLRWstate
from klrw.perfect_complex import (
    KLRWIrreducibleProjectiveModule,
    KLRWPerfectComplex,
)
from klrw.gradings import (
    HomologicalGradingGroup,
    ExtendedQuiverGradingGroup,
)


def one_term_complex(
    base_R,
    quiver: FramedDynkinDiagram_class | CartanType_abstract,
    state: KLRWstate | tuple[NodeInFramedQuiver],
    differential_degree=-1,
    sign: Callable | None = None,
    homological_grading_names: Iterable[Any] | None = None,
    homological_grading_group: HomologicalGradingGroup | None = None,
    extended_grading_group: ExtendedQuiverGradingGroup | None = None,
    **klrw_options,
):
    """
    Constructs a complex with one term.

    The KLRW algebra is determained from the input data.
    """
    quiver_data = FramedDynkinDiagram_with_dimensions.with_zero_dimensions(
        quiver=quiver
    )
    for v in state:
        quiver_data[v] += 1

    klrw = KLRWAlgebra(
        base_R=base_R,
        quiver_data=quiver_data,
        **klrw_options,
    )

    if not isinstance(state, KLRWstate):
        state = klrw.state(state)

    projectives = {0: [KLRWIrreducibleProjectiveModule(state, 0, klrw.grading_group)]}

    differential = {}

    return KLRWPerfectComplex(
        ring=klrw,
        projectives=projectives,
        differential=differential,
        differential_degree=differential_degree,
        sign=sign,
        homological_grading_names=homological_grading_names,
        homological_grading_group=homological_grading_group,
        extended_grading_group=extended_grading_group,
    )

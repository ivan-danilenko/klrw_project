from klrw.derived_functor import KLRWDerivedFunctor
from klrw.braided_projectives import BraidedProjective
from klrw.perfect_complex import KLRWIrreducibleProjectiveModule
from klrw.klrw_algebra import KLRWAlgebra


class BraidingFunctor(KLRWDerivedFunctor):
    def __init__(
        self,
        elementary_transposition: int,
    ):
        self.elementary_transposition = elementary_transposition

    def on_projectives(
        self,
        proj: KLRWIrreducibleProjectiveModule,
        domain_klrw_algebra: KLRWAlgebra,
    ):
        BP = BraidedProjective(
            state=proj.state,
            elementary_transposition=self.elementary_transposition,
            **self._options_from_domain_klrw_algebra_(domain_klrw_algebra),
        )
        eq_shift = BP.KLRW_algebra().grading_group(proj.equivariant_degree)
        return BP[0, eq_shift]

    @staticmethod
    def _options_from_domain_klrw_algebra_(
        domain_klrw_algebra: KLRWAlgebra
    ):
        options = domain_klrw_algebra.klrw_options()
        options["quiver"] = options["quiver_data"].quiver
        del options["quiver_data"]
        return options

    def on_crossings(
        self,
        i,
        state,
        domain_klrw_algebra: KLRWAlgebra
    ):
        proj = KLRWIrreducibleProjectiveModule(state=state)
        proj_image = self.on_projectives(proj, domain_klrw_algebra)
        return proj_image.elementary_crossing_to_chain_map(i)

    def on_dots(
        self,
        dot_index,
        state,
        domain_klrw_algebra: KLRWAlgebra
    ):
        proj = KLRWIrreducibleProjectiveModule(state=state)
        proj_image = self.on_projectives(proj, domain_klrw_algebra)
        return proj_image.dot_to_chain_map(dot_index)

    def on_parameters(self, coeff):
        return coeff

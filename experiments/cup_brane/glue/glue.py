"""
Glue the stable envelopes S_mu into the cup brane exactly as
klrw.standard_ebranes.standard_ebranes does (same chains, same algebra, same
shifts, same Ext computation over QQ, same cone), but without Gurobi
(see patches.py), recording every gluing map and the provenance of every
projective of the glued complex.

    sage -python glue/glue.py TYPE RANK NODE        # e.g.  A 2 1

Writes glue/data/<TYPE><RANK>_w<NODE>.json.
"""
import sys
import os
import time
import json
import itertools
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patches  # noqa: E402

patches.install()

from sage.combinat.root_system.root_system import RootSystem  # noqa: E402
from sage.combinat.root_system.cartan_type import CartanType  # noqa: E402
from sage.rings.integer_ring import ZZ  # noqa: E402
from sage.rings.rational_field import QQ  # noqa: E402
from sage.matrix.constructor import matrix  # noqa: E402

from klrw.framed_dynkin import FramedDynkinDiagram_with_dimensions, NodeInFramedQuiver  # noqa: E402
from klrw.klrw_algebra import KLRWAlgebra  # noqa: E402
from klrw.stable_envelopes import stable_envelope  # noqa: E402
from klrw.standard_ebranes import (  # noqa: E402
    weight_poset_in_minuscule_rep,
    moving_strands_from_weights,
)
from klrw.perfect_complex import KLRWPerfectComplex  # noqa: E402
from klrw.free_complex import ShiftedObject  # noqa: E402

DATA = os.path.join(HERE, "data")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# serialisation helpers
# ---------------------------------------------------------------------------
def _json_default(o):
    try:
        return int(o)
    except Exception:
        return str(o)


def hint(g):
    """element of the (rank one) homological grading group -> int"""
    return int(sum(c for _, c in g))


def node_name(node):
    return repr(node)


def state_names(state):
    return [node_name(x) for x in state]


def degree_dict(deg):
    return {repr(label): int(c) for label, c in deg}


def exp_items(exp):
    if hasattr(exp, "sparse_iter"):
        return [(int(p), int(e)) for p, e in exp.sparse_iter()]
    return [(p, int(e)) for p, e in enumerate(exp) if e]


def element_terms(KLRW, elem):
    """KLRW (or opposite-algebra) element -> list of term dicts"""
    value = getattr(elem, "value", elem)
    names = KLRW.base().variable_names()
    out = []
    for braid, coeff in value:
        for exp, c in coeff.iterator_exp_coeff():
            if c == 0:
                continue
            params, dots = {}, {}
            for p, e in exp_items(exp):
                (dots if names[p].startswith("x_") else params)[names[p]] = e
            out.append(
                dict(
                    word=[int(a) for a in braid.word()],
                    left_state=state_names(braid.left_state()),
                    right_state=state_names(braid.right_state()),
                    params=params,
                    dots=dots,
                    coeff=str(c),
                )
            )
    return out


def mono_str(params):
    return "*".join(n if p == 1 else "{}^{}".format(n, p) for n, p in sorted(params.items())) or "1"


# ---------------------------------------------------------------------------
# normal form of a gluing map inside its homotopy class
# ---------------------------------------------------------------------------
def l1_minimal_representative(hom_set, ext_basis, rep):
    """
    Among all chain maps rep + (boundary) in the same Ext class, one with
    minimal l1 norm of the coefficient vector in the monomial basis of the
    graded Hom space (exact rational LP, PPL backend).
    """
    from sage.numerical.mip import MixedIntegerLinearProgram
    from sage.modules.free_module_element import vector

    amb = hom_set.basis._ambient_homset.basis
    v = amb._vector_from_element_(rep)
    B = ext_basis._coordinate_boundary_module.basis()
    N = len(v)
    l1_before = sum(abs(c) for c in v)
    info = dict(l1_before=str(l1_before), terms_before=sum(1 for c in v if c), boundary_rank=len(B))
    if not B:
        info.update(l1_after=str(l1_before), terms_after=info["terms_before"])
        return rep, info
    p = MixedIntegerLinearProgram(solver="PPL", maximization=False)
    y = p.new_variable(real=True, nonnegative=False)
    s = p.new_variable(real=True, nonnegative=True)
    support = sorted(set(i for i, c in v.dict().items() if c) | set(i for b in B for i in b.dict()))
    for i in support:
        expr = v[i] + p.sum(b[i] * y[j] for j, b in enumerate(B) if b[i])
        p.add_constraint(s[i] - expr >= 0)
        p.add_constraint(s[i] + expr >= 0)
    p.set_objective(p.sum(s[i] for i in support))
    p.solve()
    yv = p.get_values(y)
    w = v + sum((yv[j] * b for j, b in enumerate(B)), vector(v.base_ring(), N))
    # still the same class and a cycle
    assert hom_set.basis.next_differential * w == 0
    amb_el = amb._element_from_vector_(w, check=False)
    new = hom_set._element_constructor_(amb_el, check=False)
    info.update(l1_after=str(sum(abs(c) for c in w)), terms_after=sum(1 for c in w if c))
    return new, info


# ---------------------------------------------------------------------------
# homotopy-invariant information about a gluing map
# ---------------------------------------------------------------------------
CATEGORIES = {
    "u": lambda params, dots: any(k.startswith("u_") for k in params),
    "negative_exponent": lambda params, dots: any(v < 0 for v in params.values()),
    "r_inverse": lambda params, dots: any(k.startswith("r_") and v < 0 for k, v in params.items()),
    "t_inverse": lambda params, dots: any(k.startswith("t_") and v < 0 for k, v in params.items()),
    "u_inverse": lambda params, dots: any(k.startswith("u_") and v < 0 for k, v in params.items()),
    "dots": lambda params, dots: bool(dots),
    "any_parameter": lambda params, dots: bool(params),
}


def coordinate_analysis(KLRW_QQ, hom_set, ext_basis, rep):
    """
    Coordinates of the graded Hom space = (grading, codomain index, domain index,
    braid word, exponent).  Returns
      key -> coordinate index,
      rigid: set of coordinates i with rep_i != 0 and b_i = 0 for every boundary b
             (these coefficients are the same for every representative of the class),
      removable: category -> bool  (can all terms of that category be killed by
             adding a boundary?)
    """
    from sage.matrix.constructor import matrix as mat_
    from sage.modules.free_module_element import vector

    names = KLRW_QQ.base().variable_names()
    amb = hom_set.basis._ambient_homset.basis
    keys = []
    cats = []
    for type_ in amb.types():
        gc = amb.graded_component_of_type(type_)
        for braid, exp in gc.basis(as_tuples=True):
            params, dots = {}, {}
            for p_, e in exp_items(exp):
                (dots if names[p_].startswith("x_") else params)[names[p_]] = e
            keys.append((hint(type_.grading), int(type_.codomain_index), int(type_.domain_index),
                         tuple(int(a) for a in braid.word()),
                         tuple(sorted(params.items())), tuple(sorted(dots.items()))))
            cats.append(set(c for c, f in CATEGORIES.items() if f(params, dots)))
    assert len(keys) == len(amb)
    index = {k: i for i, k in enumerate(keys)}
    v = amb._vector_from_element_(rep)
    B = ext_basis._coordinate_boundary_module.basis()
    touched = set(i for b in B for i in b.dict())
    rigid = set(i for i, c in v.dict().items() if c and i not in touched)
    removable = {}
    for cat in CATEGORIES:
        C = [i for i in range(len(keys)) if cat in cats[i]]
        fC = vector(QQ, [v[i] for i in C])
        if not fC:
            removable[cat] = None  # nothing to remove
            continue
        if not B:
            removable[cat] = False
            continue
        BC = mat_(QQ, [[b[i] for i in C] for b in B])  # rows = boundaries
        # is -fC in the row space of BC ?
        removable[cat] = bool((-fC) in BC.row_space())
    return index, rigid, removable


# ---------------------------------------------------------------------------
# the computation
# ---------------------------------------------------------------------------
def run(cartan, node, out_path=None, normal=None):
    T0 = time.time()
    timings = {}
    ct = CartanType(cartan)
    Phi = RootSystem(ct)
    lambd = Phi.weight_lattice().fundamental_weight(node)
    pos = weight_poset_in_minuscule_rep(Phi, lambd)
    pos_dual = pos.dual()

    # --- chains, as in standard_ebranes ------------------------------------
    sequences_by_weight = {}
    for mu in pos:
        right_weight_sequence = next(pos_dual.maximal_chains_iterator([mu]))
        right_sequence = tuple(moving_strands_from_weights(iter(right_weight_sequence)))
        left_weight_sequence = next(pos.maximal_chains_iterator([mu]))
        left_sequence = tuple(moving_strands_from_weights(reversed(left_weight_sequence)))
        sequences_by_weight[mu] = (left_sequence, right_sequence)

    (lam_index, _), = tuple(lambd)
    left_framing = NodeInFramedQuiver(lam_index, framing=True)
    dual_lam_index = ct.opposition_automorphism()[lam_index]
    right_framing = NodeInFramedQuiver(dual_lam_index, framing=True)

    DD = FramedDynkinDiagram_with_dimensions(ct)
    DD[left_framing] += 1
    DD[right_framing] += 1
    left_seq, right_seq = next(iter(sequences_by_weight.values()))
    for nd in left_seq + right_seq:
        DD[nd] += 1
    # Workaround: dimensions_dict is a defaultdict, so every lookup of a node
    # with dimension 0 inserts a 0 entry, and the (dataclass) equality of two
    # copies of the quiver data then depends on which lookups have happened.
    # KLRW._replace_(base_R=QQ) then has no coercion from KLRW (base_change
    # asserts).  Pre-filling every vertex makes the dict complete.
    for v in DD.quiver.vertices():
        DD.dimensions_dict[v] += 0
    KLRW = KLRWAlgebra(
        ZZ,
        DD,
        dot_scaling=True,
        edge_scaling=True,
        vertex_scaling=True,
        invertible_parameters=True,
        warnings=True,
    )
    log("algebra:", KLRW.base())

    weights = list(pos_dual.linear_extension())
    mu_index = {mu: i for i, mu in enumerate(weights)}

    # --- stable envelopes ---------------------------------------------------
    t = time.time()
    stab_ZZ = {}
    envelopes = []
    for i, mu in enumerate(weights):
        lseq, rseq = sequences_by_weight[mu]
        t1 = time.time()
        st = stable_envelope(
            KLRW, left_framing, right_framing, left_sequence=lseq, right_sequence=rseq
        )
        stab_ZZ[mu] = st
        n = len(lseq) + len(rseq)
        # vertex labels in the order stable_envelope uses (itertools.product order)
        verts = defaultdict(list)
        for eps in itertools.product(range(2), repeat=n):
            verts[sum(eps)].append(eps)
        # check the state of every vertex against our reading of the convention
        for g, projs in st.projectives().items():
            h = hint(g)
            assert len(projs) == len(verts[h])
            for eps, pr in zip(verts[h], projs):
                exp_state = (
                    [lseq[x] for x in reversed(range(len(lseq))) if eps[x] == 0]
                    + [left_framing]
                    + [lseq[x] for x in range(len(lseq)) if eps[x] == 1]
                    + [rseq[x] for x in range(len(rseq)) if eps[len(lseq) + x] == 0]
                    + [right_framing]
                    + [rseq[x] for x in reversed(range(len(rseq))) if eps[len(lseq) + x] == 1]
                )
                assert state_names(pr.state) == [node_name(x) for x in exp_state], (
                    eps, pr.state, exp_state)
        nterms = sum(
            len(element_terms(KLRW, e))
            for _, mat in st.differential
            for e in mat.dict(copy=False).values()
        )
        envelopes.append(
            dict(
                index=i,
                weight=str(mu),
                dynkin_labels={str(k): int(c) for k, c in mu},
                left_word=[int(x.node) for x in lseq],
                right_word=[int(x.node) for x in rseq],
                n_strands=n,
                n_projectives=2 ** n,
                n_differential_terms=nterms,
                seconds=time.time() - t1,
                vertices={str(h): [list(e) for e in v] for h, v in verts.items()},
            )
        )
        log("S_{} = S[{}]: left {} right {}  ({} terms, {:.1f}s)".format(
            i, mu, envelopes[-1]["left_word"], envelopes[-1]["right_word"], nterms, time.time() - t1))
    timings["stable_envelopes"] = time.time() - t

    KLRW_QQ = KLRW._replace_(base_R=QQ)
    stab = {mu: st.base_change(KLRW_QQ) for mu, st in stab_ZZ.items()}
    vert_of = {i: {int(h): [tuple(e) for e in v] for h, v in env["vertices"].items()} for i, env in enumerate(envelopes)}

    # --- gluing -------------------------------------------------------------
    e_brane = stab[weights[0]]
    # provenance: hom degree -> list of (envelope index, eps) for every projective
    prov = {}
    for g, projs in e_brane.projectives().items():
        h = hint(g)
        prov[h] = [(0, vert_of[0][h][p]) for p in range(len(projs))]

    steps = []
    failed = None
    for idx, mu in enumerate(weights[1:], 1):
        t1 = time.time()
        lseq, rseq = sequences_by_weight[mu]
        stab_mu = stab[mu]
        degree = sum(
            KLRW_QQ.grading_group.crossing_grading(lseq[i], lseq[j])
            for j in range(len(lseq))
            for i in range(j)
        )
        degree += sum(
            KLRW_QQ.grading_group.crossing_grading(lseq[i], left_framing)
            for i in range(len(lseq))
        )
        shift_h = -len(lseq) + 1
        shifted = stab_mu[shift_h, -degree]
        hom_set = shifted.hom_set(e_brane)
        ext_set = hom_set.homology()
        ext_basis = ext_set.basis
        n_ext = len(ext_basis)
        ambient = len(hom_set.basis._ambient_homset.basis)
        cycles = len(hom_set.basis)
        boundaries = ext_basis._coordinate_boundary_module.rank()
        t_ext = time.time() - t1
        step = dict(
            step=idx,
            envelope=idx,
            weight=str(mu),
            shift=dict(homological=shift_h, equivariant=str(-degree)),
            hom_dims=dict(graded_maps=ambient, cycles=cycles, boundaries=boundaries, ext=n_ext),
            seconds_ext=t_ext,
        )
        log("step {}: S[{}] -> E   Hom^0: maps {}, cycles {}, boundaries {}, Ext {}  [{:.1f}s]".format(
            idx, mu, ambient, cycles, boundaries, n_ext, t_ext))
        if n_ext != 1:
            step["error"] = "len(ext_basis) == {} (standard_ebranes asserts 1)".format(n_ext)
            steps.append(step)
            if n_ext == 0:
                failed = step["error"]
                break
        ext = ext_basis(0)
        rep = ext.representative()
        if normal == "l1":
            t3 = time.time()
            rep, info = l1_minimal_representative(hom_set, ext_basis, rep)
            info["seconds"] = time.time() - t3
            step["normal_form"] = info
            log("   l1 normal form: l1 {} -> {}, terms {} -> {} [{:.1f}s]".format(
                info["l1_before"], info["l1_after"], info["terms_before"], info["terms_after"], info["seconds"]))

        # components of the gluing map (domain: S_mu shifted, codomain: current E)
        dom = rep.parent().domain
        dom_orig, _ = ShiftedObject.original_and_shift(dom)
        assert dom_orig is stab_mu or dom_orig == stab_mu
        hs = hint(dom.hom_shift())
        cindex, rigid, removable = coordinate_analysis(KLRW_QQ, hom_set, ext_basis, rep)
        comps = []
        for grading, mat in rep:
            g = hint(grading)
            for (i, j), elem in mat.dict(copy=False).items():
                src_eps = vert_of[idx][g + hs][j]
                tgt_env, tgt_eps = prov[g][i]
                for term in element_terms(KLRW_QQ, elem):
                    ckey = (g, int(i), int(j), tuple(term["word"]),
                            tuple(sorted(term["params"].items())), tuple(sorted(term["dots"].items())))
                    term.update(
                        hom_deg_in_cone_domain=g,
                        src_envelope=idx,
                        src_eps=list(src_eps),
                        tgt_envelope=tgt_env,
                        tgt_eps=list(tgt_eps),
                        rigid=cindex[ckey] in rigid,
                    )
                    comps.append(term)
        step["map_terms"] = comps
        step["n_map_terms"] = len(comps)
        step["n_rigid_terms"] = sum(1 for t in comps if t["rigid"])
        step["removable_by_homotopy"] = removable
        log("   rigid terms {}/{}; removable by adding a boundary: {}".format(
            step["n_rigid_terms"], len(comps), {k: v for k, v in removable.items() if v is not None}))

        # the cone; provenance of its projectives
        t2 = time.time()
        new = rep.cone()
        dsh = new._domain_shifted
        dsh_orig, _ = ShiftedObject.original_and_shift(dsh)
        assert dsh_orig is stab_mu or dsh_orig == stab_mu
        hs2 = hint(dsh.hom_shift())
        new_prov = {}
        for g, projs in new.projectives().items():
            h = hint(g)
            n_dom = dsh.component_rank(g)
            entries = [(idx, vert_of[idx][h + hs2][p]) for p in range(n_dom)] + prov.get(h, [])
            assert len(entries) == len(projs), (h, len(entries), len(projs))
            new_prov[h] = entries
        # force the differential (Differential.__init__ checks d^2 = 0)
        _ = new.differential
        step["seconds_cone"] = time.time() - t2
        step["cone_size"] = sum(len(p) for p in new.projectives().values())
        steps.append(step)
        log("   gluing map: {} terms; cone has {} projectives [{:.1f}s]".format(
            len(comps), step["cone_size"], time.time() - t2))
        e_brane = new
        prov = new_prov
    timings["gluing"] = sum(s.get("seconds_ext", 0) + s.get("seconds_cone", 0) for s in steps)

    # --- dump the glued complex ----------------------------------------------
    projectives = []
    gindex = {}
    for g in sorted(e_brane.projectives().keys(), key=hint):
        h = hint(g)
        for p, pr in enumerate(e_brane.projectives(g)):
            env, eps = prov[h][p]
            gindex[h, p] = len(projectives)
            projectives.append(
                dict(
                    index=len(projectives),
                    hom_deg=h,
                    position=p,
                    state=state_names(pr.state),
                    equivariant_degree=str(pr.equivariant_degree),
                    equivariant_degree_dict=degree_dict(pr.equivariant_degree),
                    envelope=env,
                    eps=list(eps),
                )
            )
    diff_degree = hint(e_brane.differential.degree().homological_part()) if hasattr(
        e_brane.differential.degree(), "homological_part") else -1
    terms = []
    for g, mat in e_brane.differential:
        h = hint(g)
        for (i, j), entry in mat.dict(copy=False).items():
            src = gindex[h, j]
            tgt = gindex[h + diff_degree, i]
            for term in element_terms(KLRW_QQ, entry):
                term.update(source=src, target=tgt,
                            src_envelope=projectives[src]["envelope"],
                            tgt_envelope=projectives[tgt]["envelope"])
                terms.append(term)

    # all coefficients integral?  then also rebuild over ZZ as standard_ebranes does
    integral = all(QQ(t["coeff"]).denominator() == 1 for t in terms)
    zz_ok = None
    if integral and failed is None:
        t2 = time.time()
        dZZ = {}
        for g, mat in e_brane.differential:
            dZZ[g] = matrix(
                KLRW.opposite,
                ncols=mat.ncols(),
                nrows=mat.nrows(),
                entries={
                    (i, j): KLRW.sum_of_terms((braid, KLRW.base()(coeff)) for braid, coeff in entry.value)
                    for (i, j), entry in mat.dict(copy=False).items()
                },
                sparse=True,
            )
        E_ZZ = KLRWPerfectComplex(
            ring=KLRW,
            projectives=e_brane.projectives(),
            differential=dZZ,
            differential_degree=e_brane.differential.degree(),
        )
        zz_ok = True  # construction asserts d^2 = 0
        timings["rebuild_ZZ"] = time.time() - t2

    timings["total"] = time.time() - T0
    data = dict(
        case="{}{}_w{}".format(ct.type(), ct.rank(), node) + ("" if normal is None else "__" + normal),
        normal_form=normal,
        cartan_type=[ct.type(), ct.rank()],
        node=node,
        left_framing=node_name(left_framing),
        right_framing=node_name(right_framing),
        variables=list(KLRW.base().variable_names()),
        weights_in_gluing_order=[str(mu) for mu in weights],
        envelopes=envelopes,
        steps=steps,
        failed=failed,
        projectives=projectives,
        differential_degree=diff_degree,
        terms=terms,
        all_coefficients_integral=integral,
        rebuilt_over_ZZ_d2_zero=zz_ok,
        timings=timings,
        solver_stats=dict(patches.STATS),
        conventions=dict(
            eps="eps[s]=1: strand s is to the right of its red strand; strands 0..len(left)-1 are the left word "
                "(first = innermost around W_left), the rest the right word (last = innermost around W_right); "
                "state = [left eps=0 reversed] W_left [left eps=1] [right eps=0] W_right [right eps=1 reversed]",
            term="KLRW element coeff*braid, left_state = source projective, right_state = target projective; "
                 "dots indexed on the left state",
            envelope="index into weights_in_gluing_order; the glued complex is Cone(S_k[shift] -> E_{k-1}) iterated",
        ),
    )
    if out_path is None:
        os.makedirs(DATA, exist_ok=True)
        out_path = os.path.join(DATA, data["case"] + ".json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1, default=_json_default)
    log("saved", out_path)
    summarize(data)
    return data


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def summarize(data):
    P = data["projectives"]
    T = data["terms"]
    log("=" * 70)
    log("case {}: {} projectives, {} differential terms, total {:.1f}s".format(
        data["case"], len(P), len(T), data["timings"]["total"]))
    log("Ext dims per step:", [s["hom_dims"]["ext"] for s in data["steps"]])
    glue = [t for t in T if t["src_envelope"] != t["tgt_envelope"]]
    diag = [t for t in T if t["src_envelope"] == t["tgt_envelope"]]
    log("terms inside envelopes: {}, gluing terms: {}".format(len(diag), len(glue)))
    mon = Counter(mono_str(t["params"]) for t in glue)
    dots = Counter(mono_str(t["dots"]) for t in glue)
    coeff = Counter(t["coeff"] for t in glue)
    log("gluing monomials:", dict(mon))
    log("gluing dots:", dict(dots))
    log("gluing coeffs:", dict(coeff))
    log("gluing identity components (empty word, no params, no dots):",
        sum(1 for t in glue if not t["word"] and not t["params"] and not t["dots"]))
    log("gluing components with empty word:", sum(1 for t in glue if not t["word"]))
    neg = [t for t in glue if any(v < 0 for v in t["params"].values())]
    log("gluing terms with negative exponents:", len(neg))
    log("gluing terms with u:", sum(1 for t in glue if any(k.startswith("u_") for k in t["params"])))
    pairs = Counter((t["src_envelope"], t["tgt_envelope"]) for t in glue)
    log("gluing terms by (src envelope, tgt envelope):", dict(sorted(pairs.items())))


if __name__ == "__main__":
    ctype = sys.argv[1]
    rank = int(sys.argv[2])
    node = int(sys.argv[3])
    normal = sys.argv[4] if len(sys.argv) > 4 else None
    run([ctype, rank], node, normal=normal)

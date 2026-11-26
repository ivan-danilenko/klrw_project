"""
Gluing over the polynomial algebra, with the terms of every gluing map printed in poset
language, and the freedom (null-homotopic maps) made explicit.

    sage -python glue/poly_step.py CASE STEP
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glue_poly as GP
from glue_poly import *
from collections import Counter, defaultdict


def poset_names(D):
    envs = D["envelopes"]
    def names(word):
        cnt = Counter(); out = []
        for c in word:
            cnt[c] += 1; out.append((c, cnt[c]))
        return out
    canon = names(envs[-1]["left_word"] + envs[-1]["right_word"])
    elem = {x: "p%d" % (i + 1) for i, x in enumerate(canon)}
    info = {e["index"]: ([elem[x] for x in names(e["left_word"] + e["right_word"])], len(e["left_word"])) for e in envs}
    return info


def layout(info, env, eps):
    chain, nl = info[env]
    left = list(range(nl)); right = list(range(nl, len(chain)))
    pos = [chain[s] for s in reversed(left) if eps[s] == 0] + ["W"] + [chain[s] for s in left if eps[s] == 1] \
        + [chain[s] for s in right if eps[s] == 0] + ["W'"] + [chain[s] for s in reversed(right) if eps[s] == 1]
    state = {}
    for s in left: state[chain[s]] = "A" if eps[s] == 0 else "BL"
    for s in right: state[chain[s]] = "BR" if eps[s] == 0 else "C"
    return pos, state


def describe(info, env_s, eps_s, env_t, eps_t, term):
    pos_s, st_s = layout(info, env_s, eps_s); pos_t, st_t = layout(info, env_t, eps_t)
    lab = list(pos_s)
    for letter in term["word"]:
        lab[letter - 1], lab[letter] = lab[letter], lab[letter - 1]
    sigma = {a: b for a, b in zip(lab, pos_t) if a not in ("W", "W'")}
    elems = sorted(st_s, key=lambda x: int(x[1:]))
    ch = []
    for p in elems:
        q = sigma[p]
        if st_s[p] != st_t[q] or q != p:
            ch.append("%s:%s>%s%s" % (p, st_s[p], st_t[q], "" if q == p else "(%s)" % q))
    mono = " ".join("%s^%d" % kv if kv[1] != 1 else kv[0] for kv in sorted(term["params"].items())) or "1"
    dots = " ".join(term["dots"]) or "-"
    return " ".join(ch), mono, dots, " ".join("%s=%s" % (p, st_s[p]) for p in elems)


def main(case, step, normal="l1"):
    D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", case + ".json")))
    info = poset_names(D)
    # re-run the polynomial gluing up to `step`, keeping provenance
    glue_terms = [t for t in D["terms"] if t["src_envelope"] != t["tgt_envelope"]]
    resc, _ = analyze.nonneg_rescaling(D, glue_terms, D["variables"])
    ct = CartanType([D["cartan_type"][0], D["cartan_type"][1]])
    node = D["node"]
    Phi = RootSystem(ct)
    lambd = Phi.weight_lattice().fundamental_weight(node)
    pos = weight_poset_in_minuscule_rep(Phi, lambd); pos_dual = pos.dual()
    seqs = {}
    for mu in pos:
        rs = tuple(moving_strands_from_weights(iter(next(pos_dual.maximal_chains_iterator([mu])))))
        ls = tuple(moving_strands_from_weights(reversed(next(pos.maximal_chains_iterator([mu])))))
        seqs[mu] = (ls, rs)
    (lam_index, _), = tuple(lambd)
    lf = NodeInFramedQuiver(lam_index, framing=True)
    rf = NodeInFramedQuiver(ct.opposition_automorphism()[lam_index], framing=True)
    DD = FramedDynkinDiagram_with_dimensions(ct)
    DD[lf] += 1; DD[rf] += 1
    l0, r0 = next(iter(seqs.values()))
    for nd in l0 + r0: DD[nd] += 1
    for v in DD.quiver.vertices(): DD.dimensions_dict[v] += 0
    KLRW = KLRWAlgebra(ZZ, DD, dot_scaling=True, edge_scaling=True, vertex_scaling=True,
                       invertible_parameters=False, warnings=True)
    weights = list(pos_dual.linear_extension())
    stabZ = {mu: stable_envelope(KLRW, lf, rf, left_sequence=seqs[mu][0], right_sequence=seqs[mu][1]) for mu in weights}
    KQ = KLRW._replace_(base_R=QQ)
    stab = {mu: st.base_change(KQ) for mu, st in stabZ.items()}
    Gr = KQ.grading_group; Dalg = KQ.base()
    mdeg = lambda mono: Gr.sum([e * Dalg.element_degree(Dalg(x), Gr) for x, e in mono.items() if e])
    n = len(l0) + len(r0)
    vert = defaultdict(list)
    for eps in itertools.product(range(2), repeat=n):
        vert[sum(eps)].append(eps)
    e_brane = stab[weights[0]]
    prov = {G.hint(g): [(0, vert[G.hint(g)][p]) for p in range(len(projs))] for g, projs in e_brane.projectives().items()}
    for idx, mu in enumerate(weights[1:], 1):
        lseq, rseq = seqs[mu]
        degree = sum(Gr.crossing_grading(lseq[i], lseq[j]) for j in range(len(lseq)) for i in range(j))
        degree += sum(Gr.crossing_grading(lseq[i], lf) for i in range(len(lseq)))
        shift_h = -len(lseq) + 1
        shifted = stab[mu][shift_h, -degree + mdeg(resc[idx]) - mdeg(resc[0])]
        hom_set = shifted.hom_set(e_brane)
        ext_basis = hom_set.homology().basis
        rep = ext_basis(0).representative()
        cycles = len(hom_set.basis); bnd = ext_basis._coordinate_boundary_module.rank()
        if normal == "l1" and bnd:
            rep, _ = G.l1_minimal_representative(hom_set, ext_basis, rep)
        dom = rep.parent().domain
        hs = G.hint(dom.hom_shift())
        if idx == step:
            print("step %d: chain maps %d, null-homotopic %d; representative with fewest terms:" % (idx, cycles, bnd))
            shapes = Counter(); examples = {}
            for grading, mat in rep:
                g = G.hint(grading)
                for (i, j), elem in mat.dict(copy=False).items():
                    src_eps = vert[g + hs][j]
                    tgt_env, tgt_eps = prov[g][i]
                    for term in G.element_terms(KQ, elem):
                        ch, mono, dots, st = describe(info, idx, src_eps, tgt_env, tgt_eps, term)
                        key = ("S%d" % tgt_env, ch, mono, dots, term["coeff"].lstrip("-"))
                        shapes[key] += 1; examples.setdefault(key, st)
            for key, v in sorted(shapes.items(), key=lambda kv: (kv[0][0], -kv[1])):
                print("   x%-2d -> %s  |c|=%s  [%s]  dots %-6s %-22s e.g. %s" % (v, key[0], key[4], key[1], key[3], key[2], examples[key]))
            return
        new = rep.cone()
        dsh = new._domain_shifted
        hs2 = G.hint(dsh.hom_shift())
        new_prov = {}
        for g, projs in new.projectives().items():
            h = G.hint(g)
            nd = dsh.component_rank(g)
            new_prov[h] = [(idx, vert[h + hs2][p]) for p in range(nd)] + prov.get(h, [])
        _ = new.differential
        e_brane = new; prov = new_prov


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))

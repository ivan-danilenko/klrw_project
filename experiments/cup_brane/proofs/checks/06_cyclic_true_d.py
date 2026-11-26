"""
True (order-by-order corrected, complete ansatz) differential of a small stack, saved as JSON
in proofs/checks/06_data/ (format of cupbrane.run_case).

    sage -python proofs/checks/06_cyclic_true_d.py TYPE RANK RED WORD
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cupbrane import run_case

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3]); word = tuple(int(c) for c in sys.argv[4])
    cube, d, log, data = run_case(ct, [(red, word)], out_dir=os.path.join(HERE, "06_data"))
    dots = [t for t in data["terms"] if t["dots"]]
    print("terms", len(data["terms"]), "with dots", len(dots), "max |coeff|", max(abs(t["coeff"]) for t in data["terms"]))

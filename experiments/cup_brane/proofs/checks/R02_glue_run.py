"""
Referee check R02 (independent of the 02_* scripts): run the real gluing procedure
(glue/glue.py = standard_ebranes step by step with klrw's own Hom/Ext machinery) for
sl_{n+1}, omega_1 and write the dump to a directory OUTSIDE the repository data.

    sage -python proofs/checks/R02_glue_run.py N OUTDIR

The dump is then compared with the closed formula by R02_compare_glue.py.
Nothing under glue/ is modified (glue.run is called with an explicit out_path).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "glue"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import glue  # noqa: E402

if __name__ == "__main__":
    n = int(sys.argv[1]); out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    glue.run(["A", n], 1, out_path=os.path.join(out, "A%d_w1.json" % n))

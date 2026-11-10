"""Compatibility shim: lookup tables now live in biotoolkit/sequence.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from biotoolkit.sequence import (  # noqa: E402,F401
    DNA_CODONS as DNA_Codons,
    DNA_REVERSE_COMPLEMENT as DNA_ReverseComplement,
    NUCLEOTIDES as Nucleotides,
)

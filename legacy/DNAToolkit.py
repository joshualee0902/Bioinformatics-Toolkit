"""Compatibility shim for the legacy Tkinter app.

The real implementations now live in biotoolkit/sequence.py. This file keeps
the old camelCase names so legacy/functions.py runs unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from biotoolkit.sequence import (  # noqa: E402,F401
    all_proteins_from_orfs,
    codon_usage,
    count_nuc_frequency as countNucFrequency,
    gc_content,
    gc_content_subsec,
    gen_reading_frames,
    proteins_from_rf,
    reverse_complement,
    transcription,
    translate_seq,
    validate_seq as validateSeq,
)

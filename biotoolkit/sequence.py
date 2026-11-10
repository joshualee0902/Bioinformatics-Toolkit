"""Core DNA sequence utilities.

These are the original toolkit functions (from the Tkinter app's DNAToolkit.py
and structures.py), refactored with type hints, docstrings, and fixes for the
edge cases that used to crash. They are reused as the feature-engineering layer.

Ambiguous base rule: "N" (unknown base) is the only non-ACGT character allowed,
and only when a function says so. N is never counted as G or C, and any codon
that contains an N translates to "X" (unknown amino acid).
"""

from __future__ import annotations

from collections import Counter

NUCLEOTIDES = ["A", "C", "G", "T"]
AMBIGUOUS_BASE = "N"
DNA_REVERSE_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
STOP = "_"
START = "M"
UNKNOWN_AA = "X"

DNA_CODONS = {
    # 'M' - START, '_' - STOP
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TGT": "C", "TGC": "C",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TTT": "F", "TTC": "F",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "CAT": "H", "CAC": "H",
    "ATA": "I", "ATT": "I", "ATC": "I",
    "AAA": "K", "AAG": "K",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATG": "M",
    "AAT": "N", "AAC": "N",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TGG": "W",
    "TAT": "Y", "TAC": "Y",
    "TAA": "_", "TAG": "_", "TGA": "_",
}  # fmt: skip

# Sorted list of the 64 codons, used as a fixed feature order.
CODONS = sorted(DNA_CODONS)
STOP_CODONS = [codon for codon, aa in DNA_CODONS.items() if aa == STOP]


def validate_seq(dna_seq: str, allow_n: bool = False) -> str | bool:
    """Return the uppercased sequence if it is valid DNA, otherwise False.

    Valid means every character is A, C, G or T (plus N when allow_n=True).
    An empty string is not a valid sequence.
    """
    tmpseq = dna_seq.upper()
    allowed = set(NUCLEOTIDES) | ({AMBIGUOUS_BASE} if allow_n else set())
    if not tmpseq or any(nuc not in allowed for nuc in tmpseq):
        return False
    return tmpseq


def count_nuc_frequency(seq: str) -> dict[str, int]:
    """Count A, C, G and T in a sequence. N bases are ignored.

    Raises ValueError for any other character.
    """
    freq = {"A": 0, "C": 0, "G": 0, "T": 0}
    for nuc in seq:
        if nuc in freq:
            freq[nuc] += 1
        elif nuc != AMBIGUOUS_BASE:
            raise ValueError(f"Invalid nucleotide: {nuc!r}")
    return freq


def transcription(seq: str) -> str:
    """DNA -> RNA transcription: replace thymine (T) with uracil (U)."""
    return seq.replace("T", "U")


def reverse_complement(seq: str) -> str:
    """Swap A<->T and G<->C, then reverse the string. N stays N."""
    return "".join(DNA_REVERSE_COMPLEMENT[nuc] for nuc in seq)[::-1]


def gc_fraction(seq: str) -> float:
    """Fraction of G + C among the unambiguous (non-N) bases, between 0 and 1.

    Returns 0.0 for an empty sequence or one made only of N.
    """
    n_known = len(seq) - seq.count(AMBIGUOUS_BASE)
    if n_known == 0:
        return 0.0
    return (seq.count("C") + seq.count("G")) / n_known


def gc_content(seq: str) -> int:
    """GC content as a whole-number percentage (0 for an empty sequence)."""
    return round(gc_fraction(seq) * 100)


def gc_content_subsec(seq: str, k: int = 20) -> list[int]:
    """GC content percentage of each consecutive, non-overlapping window of length k.

    A trailing window shorter than k is dropped. Raises ValueError if k < 1.
    """
    if k < 1:
        raise ValueError(f"Window size k must be >= 1, got {k}")
    return [gc_content(seq[i : i + k]) for i in range(0, len(seq) - k + 1, k)]


def translate_seq(seq: str, init_pos: int = 0) -> list[str]:
    """Translate DNA into a list of amino acids, starting at init_pos.

    Codons containing a non-ACGT character (such as N) become "X".
    """
    return [
        DNA_CODONS.get(seq[pos : pos + 3], UNKNOWN_AA)
        for pos in range(init_pos, len(seq) - 2, 3)
    ]


def codon_usage(seq: str, aminoacid: str) -> dict[str, float]:
    """Relative frequency of each codon that encodes `aminoacid` in frame 0.

    Returns {} when the amino acid is valid but does not occur.
    Raises ValueError for an unknown amino acid or an invalid DNA sequence.
    """
    if aminoacid not in set(DNA_CODONS.values()):
        raise ValueError(f"Unknown amino acid: {aminoacid!r}")
    if validate_seq(seq, allow_n=True) is False:
        raise ValueError("Sequence must contain only A, C, G, T or N")

    seq = seq.upper()
    matching = [
        seq[i : i + 3]
        for i in range(0, len(seq) - 2, 3)
        if DNA_CODONS.get(seq[i : i + 3]) == aminoacid
    ]
    freq = Counter(matching)
    total_weight = sum(freq.values())
    return {codon: round(count / total_weight, 2) for codon, count in freq.items()}


def gen_reading_frames(seq: str) -> list[list[str]]:
    """Translate all six reading frames: three forward, three on the reverse complement."""
    rc = reverse_complement(seq)
    return [translate_seq(s, frame) for s in (seq, rc) for frame in range(3)]


def proteins_from_rf(aa_seq: list[str]) -> list[str]:
    """Find every protein (M ... up to a stop) in one translated reading frame.

    Nested start codons each give their own protein. Proteins with no stop
    codon before the end of the frame are not returned.
    """
    current_prot: list[str] = []
    proteins: list[str] = []
    for aa in aa_seq:
        if aa == STOP:
            proteins.extend(current_prot)
            current_prot = []
        else:
            if aa == START:
                current_prot.append("")
            current_prot = [p + aa for p in current_prot]
    return proteins


def all_proteins_from_orfs(
    seq: str, start_read_pos: int = 0, end_read_pos: int = 0, ordered: bool = False
) -> list[str]:
    """All proteins found in the six reading frames of seq (or of a slice of it).

    If end_read_pos > start_read_pos only seq[start_read_pos:end_read_pos] is read.
    With ordered=True the proteins are sorted longest first.
    """
    if end_read_pos > start_read_pos:
        seq = seq[start_read_pos:end_read_pos]
    res = [p for rf in gen_reading_frames(seq) for p in proteins_from_rf(rf)]
    if ordered:
        return sorted(res, key=len, reverse=True)
    return res

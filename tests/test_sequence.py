import pytest

from biotoolkit import sequence as s


def test_validate_seq_uppercases_valid_dna():
    assert s.validate_seq("acgt") == "ACGT"


def test_validate_seq_rejects_invalid_and_empty():
    assert s.validate_seq("ACGX") is False
    assert s.validate_seq("") is False


def test_validate_seq_n_only_when_allowed():
    assert s.validate_seq("ACNT") is False
    assert s.validate_seq("acnt", allow_n=True) == "ACNT"


def test_count_nuc_frequency():
    assert s.count_nuc_frequency("AACGTTTN") == {"A": 2, "C": 1, "G": 1, "T": 3}


def test_count_nuc_frequency_rejects_invalid():
    with pytest.raises(ValueError):
        s.count_nuc_frequency("ACGZ")


def test_transcription():
    assert s.transcription("ATGCTT") == "AUGCUU"


def test_reverse_complement():
    assert s.reverse_complement("ATGCN") == "NGCAT"
    assert s.reverse_complement("") == ""


def test_gc_content_and_fraction():
    assert s.gc_content("GGCCAATT") == 50
    assert s.gc_fraction("GCNN") == 1.0  # N is excluded from the denominator


def test_gc_content_empty_input_does_not_divide_by_zero():
    assert s.gc_content("") == 0
    assert s.gc_fraction("NNN") == 0.0


def test_gc_content_subsec():
    assert s.gc_content_subsec("GGGGAAAACC", k=4) == [100, 0]  # trailing "CC" is dropped
    assert s.gc_content_subsec("ACGT", k=10) == []


@pytest.mark.parametrize("k", [0, -1])
def test_gc_content_subsec_rejects_bad_window(k):
    with pytest.raises(ValueError):
        s.gc_content_subsec("ACGT", k=k)


def test_translate_seq():
    assert s.translate_seq("ATGGCCTAA") == ["M", "A", "_"]
    assert s.translate_seq("AATGGCC", init_pos=1) == ["M", "A"]


def test_translate_seq_handles_n():
    assert s.translate_seq("ATGNNNTAA") == ["M", "X", "_"]


def test_codon_usage():
    assert s.codon_usage("CTGCTGTTAAAA", "L") == {"CTG": 0.67, "TTA": 0.33}
    assert s.codon_usage("AAAAAA", "L") == {}


def test_codon_usage_rejects_invalid_input():
    with pytest.raises(ValueError):
        s.codon_usage("ACGT", "Z")
    with pytest.raises(ValueError):
        s.codon_usage("ACGQQQ", "L")


def test_gen_reading_frames():
    frames = s.gen_reading_frames("ATGAAATAG")
    assert len(frames) == 6
    assert frames[0] == ["M", "K", "_"]
    assert frames[3] == s.translate_seq("CTATTTCAT")


def test_proteins_from_rf():
    assert s.proteins_from_rf(["M", "A", "M", "K", "_", "M", "Q"]) == ["MAMK", "MK"]
    assert s.proteins_from_rf(["A", "_"]) == []


def test_all_proteins_from_orfs():
    seq = "ATGAAATAGCCC"
    assert s.all_proteins_from_orfs(seq) == ["MK"]
    assert s.all_proteins_from_orfs("ATGTAAATGAAACCCTAA", ordered=True) == ["MKP", "M"]
    assert s.all_proteins_from_orfs("CCCATGTAA", start_read_pos=3, end_read_pos=9) == ["M"]

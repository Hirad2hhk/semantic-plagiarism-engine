import os
import tempfile

import pytest

from src.plagiarism_engine.lsh import LSH
from src.plagiarism_engine.minhash import MinHashEngine
from src.plagiarism_engine.preprocessing import (
    generate_shingles,
    preprocess_text,
    tokenize,
)
from src.plagiarism_engine.simhash import SimHashEngine


def test_preprocessing():
    """Test text preprocessing functionality"""
    # Test basic preprocessing
    text = "Hello, World! This is a test."
    tokens = preprocess_text(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert "hello" in tokens  # Should be lowercased

    # Test empty input
    empty_tokens = preprocess_text("")
    assert empty_tokens == []

    # Test with non-string input
    non_string_tokens = preprocess_text(123)
    assert non_string_tokens == []


def test_tokenization():
    """Test tokenization with stop words removal"""
    text = "This is a test document with some stopwords"
    tokens = tokenize(text)

    assert isinstance(tokens, list)
    # Stopwords should be removed
    assert "this" not in tokens
    assert "is" not in tokens
    assert "a" not in tokens
    assert "with" not in tokens

    # Content words should remain
    assert "test" in tokens
    assert "document" in tokens


def test_shingle_generation():
    """Test shingle generation functionality"""
    tokens = ["this", "is", "a", "test", "document"]

    # Test with k=3
    shingles = generate_shingles(tokens, k=3)
    assert (
        len(shingles) == 3
    )  # Should have 3 shingles: (this, is, a), (is, a, test), (a, test, document)

    # Test with k=5 (should return 1 shingle)
    shingles_5 = generate_shingles(tokens, k=5)
    assert len(shingles_5) == 1

    # Test with k=6 (should return 0 shingles)
    shingles_6 = generate_shingles(tokens, k=6)
    assert len(shingles_6) == 0

    # Test with k=0 (should return 0 shingles)
    shingles_0 = generate_shingles(tokens, k=0)
    assert len(shingles_0) == 0


def test_minhash_signature():
    """Test MinHash signature generation"""
    # Create a simple test case
    shingles = [("test", "document"), ("document", "example")]

    minhash_engine = MinHashEngine()

    # Should generate 128-bit signature
    signature = minhash_engine.compute_signature(shingles)
    assert isinstance(signature, list)
    assert len(signature) == 128

    # Test with empty shingles
    empty_signature = minhash_engine.compute_signature([])
    assert isinstance(empty_signature, list)
    assert len(empty_signature) == 128


def test_minhash_similarity():
    """Test MinHash similarity estimation"""
    shingles1 = [("test", "document"), ("document", "example")]
    shingles2 = [("test", "document"), ("different", "example")]

    minhash_engine = MinHashEngine()

    sig1 = minhash_engine.compute_signature(shingles1)
    sig2 = minhash_engine.compute_signature(shingles2)

    similarity = minhash_engine.estimate_similarity(sig1, sig2)
    assert isinstance(similarity, float)
    assert 0.0 <= similarity <= 1.0


def test_lsh_functionality():
    """Test LSH bucketing functionality"""
    # Create an LSH instance
    lsh = LSH(threshold=0.3,signature_len=128)

    # Test adding documents
    signature1 = [1] * 128  # 2 bands of 4 rows each
    signature2 = [1] * 128  # Same signature for testing

    lsh.add_document("doc1", signature1)
    lsh.add_document("doc2", signature2)

    # Get candidate pairs (should find matches due to same signature)
    candidates = lsh.get_candidate_pairs()

    # For this simple test, we expect at least one candidate pair if they're hashed to same bucket
    # This is a basic check - actual implementation needs more comprehensive testing


def test_simhash_fingerprint():
    """Test SimHash fingerprint generation"""
    tokens = ["test", "document", "example"]

    simhash_engine = SimHashEngine()

    fingerprint = simhash_engine.compute_fingerprint(tokens)
    assert isinstance(fingerprint, int)
    # Should be a 64-bit integer (0 to 2^64 - 1)
    assert 0 <= fingerprint < 2**64


def test_simhash_hamming_distance():
    """Test SimHash Hamming distance calculation"""
    tokens1 = ["test", "document"]
    tokens2 = ["test", "example"]

    simhash_engine = SimHashEngine()

    fp1 = simhash_engine.compute_fingerprint(tokens1)
    fp2 = simhash_engine.compute_fingerprint(tokens2)

    distance = simhash_engine.hamming_distance(fp1, fp2)
    assert isinstance(distance, float)
    assert 0.0 <= distance <= 1.0


def test_edge_cases():
    """Test edge cases"""
    # Test preprocessing with various inputs
    assert preprocess_text(None) == []
    assert preprocess_text(123) == []

    # Test shingle generation with edge cases
    assert generate_shingles([], k=2) == []
    assert generate_shingles(["a"], k=2) == []

    # Test minhash with edge cases
    minhash_engine = MinHashEngine()
    assert minhash_engine.estimate_similarity([], []) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

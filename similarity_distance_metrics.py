"""
Similarity and Distance Metrics
Implements 7 metrics using Python's standard library.
"""

from math import sqrt


# ==========================================
# 1. VECTOR METRICS
# ==========================================

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))

    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    return dot / (norm_a * norm_b)


def l2_distance(a, b):
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def l1_distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))


# ==========================================
# 2. SET METRICS
# ==========================================

def jaccard_similarity(a, b):
    intersection = len(a & b)
    union = len(a | b)

    if union == 0:
        return 1.0

    return intersection / union


def dice_similarity(a, b):
    intersection = len(a & b)
    total = len(a) + len(b)

    if total == 0:
        return 1.0

    return (2 * intersection) / total


# ==========================================
# 3. HAMMING DISTANCE
# ==========================================

def hamming_distance(a, b):
    if len(a) != len(b):
        raise ValueError("Sequences must have equal lengths.")

    return sum(x != y for x, y in zip(a, b))


# ==========================================
# 4. DEMONSTRATION
# ==========================================

def main():

    # Numerical vectors
    vector_a = [2, 3]
    vector_b = [5, 7]

    print("=" * 45)
    print("VECTOR METRICS")
    print("=" * 45)

    print("Vector A:", vector_a)
    print("Vector B:", vector_b)

    print("Cosine similarity:", round(cosine_similarity(vector_a, vector_b), 4))
    print("L2 distance:       ", l2_distance(vector_a, vector_b))
    print("L1 distance:       ", l1_distance(vector_a, vector_b))
    print("Dot product:       ", dot_product(vector_a, vector_b))


    # Sets
    set_a = {"apple", "banana", "mango"}
    set_b = {"apple", "banana", "orange"}

    print("\n" + "=" * 45)
    print("SET METRICS")
    print("=" * 45)

    print("Set A:", set_a)
    print("Set B:", set_b)

    print("Jaccard similarity:", round(jaccard_similarity(set_a, set_b), 4))
    print("Dice similarity:   ", round(dice_similarity(set_a, set_b), 4))


    # Binary sequences
    sequence_a = [1, 0, 1, 1]
    sequence_b = [1, 1, 0, 1]

    print("\n" + "=" * 45)
    print("HAMMING DISTANCE")
    print("=" * 45)

    print("Sequence A:", sequence_a)
    print("Sequence B:", sequence_b)

    print("Hamming distance:", hamming_distance(sequence_a, sequence_b))


if __name__ == "__main__":
    main()
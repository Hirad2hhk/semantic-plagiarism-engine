import math
import re
from collections import Counter
from typing import List

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "will",
    "with",
    "i",
    "you",
    "we",
    "they",
    "she",
    "this",
    "these",
    "those",
    "but",
    "or",
    "not",
    "have",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "must",
    "shall",
    "were",
    "been",
    "being",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "any",
    "are",
    "aren't",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def preprocess_text(text: str) -> List[str]:
    """Perform text cleaning

    Args:
        text: Raw input text

    Returns:
        list: cleaned tokens
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [t.strip() for t in text.split() if t.strip()]

    return tokens[:50_000]  


def tokenize(text: str) -> List[str]:
    """Tokenize text and remove stopwords.

    Args:
        text (str): Input text

    Returns:
        list: List of tokens with stopwords removed
    """
    tokens = preprocess_text(text)

    
    filtered_tokens = [token for token in tokens if token not in STOPWORDS]

    return filtered_tokens


def generate_shingles(tokens: List[str], k=3) -> List[tuple]:
    """Generate overlapping word n-grams (shingles).

    Args:
        tokens: Tokenized text from `preprocess_text`
        k (int): N-gram size

    Returns:
        list[tuple]: List of shingle tuples
    """
    if len(tokens) < k or k <= 0:
        return []

    shingles = []
    for i in range(len(tokens) - k + 1):
        shingle = tuple(tokens[i : i + k])
        shingles.append(shingle)

    return shingles


def compute_tf_idf(tokens: List[str], documents: List[List[str]]) -> dict:
    """Compute TF-IDF vectors for the given tokens.

    Args:
        tokens: List of tokens to compute TF-IDF for
        documents: List of tokenized documents

    Returns:
        dict: TF-IDF values for each token
    """
    tf = Counter(tokens)

    idf = {}
    N = len(documents)

    for term in tf:
        containing_docs = sum(1 for doc in documents if term in doc)
        idf[term] = math.log(N / containing_docs) if containing_docs > 0 else 0

    tf_idf = {term: tf[term] * idf[term] for term in tf}

    return tf_idf

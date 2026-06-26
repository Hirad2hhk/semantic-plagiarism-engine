import json
import os

import click
import pandas as pd

from src.plagiarism_engine.dataset import load_corpus_directory
from src.plagiarism_engine.lsh import LSH
from src.plagiarism_engine.minhash import MinHashEngine
from src.plagiarism_engine.preprocessing import generate_shingles, tokenize
from src.plagiarism_engine.simhash import SimHashEngine


@click.group()
def cli():
    """Plagiarism Detection CLI"""


@cli.command("compare")
@click.option("--file-a", required=True, help="Path to first text file")
@click.option("--file-b", required=True, help="Path to second text file")
@click.option("--output", default="-", help="Output results JSON (- for stdout)")
def compare(file_a, file_b, output):
    """Compare two text files"""

    # Read the content of both files
    with open(file_a, "r", encoding="utf-8") as f:
        content_a = f.read()

    with open(file_b, "r", encoding="utf-8") as f:
        content_b = f.read()

    # Preprocess and tokenize
    tokens_a = tokenize(content_a)
    tokens_b = tokenize(content_b)

    # Generate shingles
    shingles_a = generate_shingles(tokens_a, k=3)
    shingles_b = generate_shingles(tokens_b, k=3)

    # Compute MinHash signatures
    minhash_engine = MinHashEngine()
    sig_a = minhash_engine.compute_signature(shingles_a)
    sig_b = minhash_engine.compute_signature(shingles_b)

    # Estimate similarity
    estimated_similarity = minhash_engine.estimate_similarity(sig_a, sig_b)

    # Create results
    results = {
        "file_a": file_a,
        "file_b": file_b,
        "estimated_similarity": estimated_similarity,
        "shingles_a_count": len(shingles_a),
        "shingles_b_count": len(shingles_b),
    }

    # Output results
    if output == "-":
        print(json.dumps(results, indent=2))
    else:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)

        # Also print the results to the terminal for visibility
        click.echo(json.dumps(results, indent=2))
        click.echo(f"Results saved to {output}")


@cli.command("corpus")
@click.option("--data", required=True, help="Directory of text files")
@click.option(
    "--threshold",
    type=float,
    default=0.8,
    help="Similarity threshold for candidate pairs",
)
@click.option("--shingle-size", type=int, default=3, help="Shingle window size")
@click.option(
    "--output", required=True, help="CSV output file path for candidate pairs"
)
def corpus(data, threshold, shingle_size, output):
    """Analyze entire text corpus"""

    # Load all documents from directory
    docs = load_corpus_directory(data)
    click.echo(f"Loaded {len(docs)} documents")

    # Create MinHash engine
    minhash_engine = MinHashEngine()

    # Build LSH structure
    lsh = LSH(threshold=threshold, signature_len=128)
    # Process each document
    doc_signatures = {}
    debug_count = 0
    for doc_id, content in docs.items():
        tokens = tokenize(content)
        shingles = generate_shingles(tokens, k=shingle_size)
        if debug_count < 3:
            click.echo(f"\n--- DEBUGGING FOR {doc_id} ---")
            click.echo(f"Raw character length: {len(content)}")
            click.echo(f"Tokens generated: {len(tokens)} -> Sample: {tokens[:5]}")
            click.echo(f"Shingles generated: {len(shingles)}")
            signature = minhash_engine.compute_signature(shingles)
            click.echo(f"Signature sample: {signature[:5]}")
            debug_count += 1
        else:
            # Compute signature
            signature = minhash_engine.compute_signature(shingles)
        doc_signatures[doc_id] = signature

        # Add to LSH structure
        lsh.add_document(doc_id, signature)

    # Find candidate pairs
    candidate_pairs = lsh.get_candidate_pairs()

    # Write to CSV (this is where the issue was - we weren't writing any actual data)
    if candidate_pairs:
        results = []
        for pair in candidate_pairs:
            doc_a, doc_b, sim = pair
            results.append({"doc_a": doc_a, "doc_b": doc_b, "similarity": sim})

        df = pd.DataFrame(results)
        # Ensure directory exists
        os.makedirs(os.path.dirname(output), exist_ok=True)
        df.to_csv(output, index=False)
        click.echo(f"Found {len(results)} candidate pairs, saved to {output}")
    else:
        click.echo("No candidate pairs found above threshold.")
        # Create an empty CSV file with headers
        df = pd.DataFrame(columns=["doc_a", "doc_b", "similarity"])
        # Ensure directory exists
        os.makedirs(os.path.dirname(output), exist_ok=True)
        df.to_csv(output, index=False)


@cli.command("pairs")
@click.option("--pairs", required=True, help="CSV file with text pairs and labels")
@click.option("--text-col-a", default="text_a", help="Column name for first text field")
@click.option(
    "--text-col-b", default="text_b", help="Column name for second text field"
)
@click.option(
    "--label-col", default="label", help="Column with ground truth labels (0/1)"
)
@click.option(
    "--limit", type=int, default=None, help="Maximum number of records to evaluate"
)
@click.option("--output", required=True, help="Output CSV with evaluation results")
def pairs(pairs, text_col_a, text_col_b, label_col, limit, output):
    """Evaluate pairs dataset"""

    # Load the CSV file
    df = pd.read_csv(pairs)

    if limit is not None:
        df = df.sample(limit, random_state=42)

    # Initialize metrics tracking
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Process each pair
    results = []

    for idx, row in df.iterrows():
        text_a = row[text_col_a]
        text_b = row[text_col_b]
        true_label = int(row[label_col])

        # Preprocess and tokenize
        tokens_a = tokenize(text_a)
        tokens_b = tokenize(text_b)

        # Generate shingles
        shingles_a = generate_shingles(tokens_a, k=3)
        shingles_b = generate_shingles(tokens_b, k=3)

        # Compute MinHash signatures
        minhash_engine = MinHashEngine()
        sig_a = minhash_engine.compute_signature(shingles_a)
        sig_b = minhash_engine.compute_signature(shingles_b)

        # Estimate similarity
        estimated_similarity = minhash_engine.estimate_similarity(sig_a, sig_b)

        # Determine prediction (assuming threshold of 0.7 for similarity)
        predicted_label = 1 if estimated_similarity >= 0.7 else 0

        # Update metrics
        if true_label == 1 and predicted_label == 1:
            true_positives += 1
        elif true_label == 0 and predicted_label == 1:
            false_positives += 1
        elif true_label == 0 and predicted_label == 0:
            true_negatives += 1
        elif true_label == 1 and predicted_label == 0:
            false_negatives += 1

        results.append(
            {
                "index": idx,
                "text_a": text_a[:50] + "..." if len(text_a) > 50 else text_a,
                "text_b": text_b[:50] + "..." if len(text_b) > 50 else text_b,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "similarity": estimated_similarity,
            }
        )

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Save detailed results
    df_results = pd.DataFrame(results)

    # Ensure directory exists for output file
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df_results.to_csv(output, index=False)

    # Save metrics to a separate file
    metrics_output = "outputs/metrics.csv"
    metrics_df = pd.DataFrame(
        [
            {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
            }
        ]
    )

    # Ensure directory exists for metrics file
    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
    metrics_df.to_csv(metrics_output, index=False)

    click.echo(f"Evaluation complete. Results saved to {output}")
    click.echo(f"Metrics saved to {metrics_output}")
    click.echo(
        f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1_score:.3f}"
    )

if __name__ == "__main__":
    cli()
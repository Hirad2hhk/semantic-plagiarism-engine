import json
import os
import time
from typing import Dict, List
import click
import pandas as pd
import numpy as np

from plagiarism_engine.dataset import load_corpus_directory
from plagiarism_engine.lsh import LSH
from plagiarism_engine.minhash import MinHashEngine
from plagiarism_engine.preprocessing import generate_shingles, tokenize
from plagiarism_engine.simhash import SimHashEngine


def compute_tfidf_weights(docs_content: List[str]) -> Dict[str, float]:
    """
    Helper function to calculate Inverse Document Frequency (IDF) weights 
    for words across a provided collection of text documents.
    """
    doc_frequencies = {}
    total_docs = len(docs_content)
    
    if total_docs == 0:
        return {}

    for content in docs_content:
        tokens = set(tokenize(content))
        for token in tokens:
            doc_frequencies[token] = doc_frequencies.get(token, 0) + 1

    
    tfidf_weights = {}
    for token, df in doc_frequencies.items():
        tfidf_weights[token] = float(np.log(total_docs / df) + 1.0)
        
    return tfidf_weights


@click.group()
def cli():
    """Plagiarism and Near-Duplicate Detection Engine CLI """
    pass


@cli.command("train-weights")
@click.option("--data", required=True, help="Directory containing the background corpus ")
@click.option("--output", default="data/processed/tfidf_weights.json", help="Path to save vocabulary weights")
def train_weights(data, output):
    """
    [EXTRA COMMAND] Pre-calculates corpus-wide IDF weights from a raw directory 
    to drive high-fidelity TF-IDF weighting inside the SimHash pipeline.
    """
    click.echo(f"Scanning background corpus at: {data}...")
    docs = load_corpus_directory(data)
    click.echo(f"Loaded {len(docs)} files. Extracting tokens and compiling global vocabulary frequencies...")
    
    contents = list(docs.values())
    weights = compute_tfidf_weights(contents)
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
        
    click.echo(f"Successfully calculated weights for {len(weights)} terms. Saved to {output}")


@cli.command("compare")
@click.option("--file-a", required=True, help="Path to first text file")
@click.option("--file-b", required=True, help="Path to second text file")
@click.option("--weights-path", default=None, help="Optional path to precomputed tfidf json weights")
@click.option("--output", default="-", help="Output results JSON (- for stdout)")
def compare(file_a, file_b, weights_path, output):
    """Compare two individual text files using both Path 1 and Path 2 algorithms """

    with open(file_a, "r", encoding="utf-8", errors="ignore") as f:
        content_a = f.read()
    with open(file_b, "r", encoding="utf-8", errors="ignore") as f:
        content_b = f.read()

    tokens_a = tokenize(content_a)
    tokens_b = tokenize(content_b)

    shingles_a = generate_shingles(tokens_a, k=3)
    shingles_b = generate_shingles(tokens_b, k=3)

    # ---- PATH 1: MINHASH ----
    minhash_engine = MinHashEngine()
    sig_a = minhash_engine.compute_signature(shingles_a)
    sig_b = minhash_engine.compute_signature(shingles_b)
    minhash_sim = minhash_engine.estimate_similarity(sig_a, sig_b)

    # ---- PATH 2: TF-IDF SIMHASH ----
    tfidf_weights = None
    if weights_path and os.path.exists(weights_path):
        with open(weights_path, "r", encoding="utf-8") as f:
            tfidf_weights = json.load(f)
            
    simhash_engine = SimHashEngine(tfidf_weights=tfidf_weights)
    fp_a = simhash_engine.compute_fingerprint(tokens_a)
    fp_b = simhash_engine.compute_fingerprint(tokens_b)
    
    
    h_dist = simhash_engine.hamming_distance(fp_a, fp_b)
    simhash_sim = 1.0 - h_dist

    results = {
        "metadata": {
            "file_a": file_a,
            "file_b": file_b,
            "tokens_a": len(tokens_a),
            "tokens_b": len(tokens_b)
        },
        "path1_minhash": {
            "estimated_jaccard": round(minhash_sim, 4)
        },
        "path2_simhash": {
            "hamming_distance": round(h_dist, 4),
            "cosine_similarity_estimate": round(simhash_sim, 4)
        }
    }

    if output == "-":
        print(json.dumps(results, indent=2))
    else:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(json.dumps(results, indent=2))
        click.echo(f"Results saved to {output}")


@cli.command("corpus")
@click.option("--data", required=True, help="Directory of text files ")
@click.option("--threshold", type=float, default=0.25, help="Similarity threshold ")
@click.option("--shingle-size", type=int, default=3, help="Shingle size ")
@click.option("--engine", type=click.Choice(['minhash', 'simhash', 'both']), default='minhash', help="Which engine pipeline to use")
@click.option("--weights-path", default="data/processed/tfidf_weighs.json", help="Path to json file holding TF-IDF lookup data")
@click.option("--output", required=True, help="CSV output file path ")
def corpus(data, threshold, shingle_size, engine, weights_path, output):
    """Analyze entire directory to map candidate matches """
    docs = load_corpus_directory(data)
    click.echo(f"Loaded {len(docs)} documents ")

    
    tfidf_weights = None
    if engine in ['simhash', 'both']:
        if weights_path and os.path.exists(weights_path):
            with open(weights_path, "r", encoding="utf-8") as f:
                tfidf_weights = json.load(f)
        else:
            click.echo("Weights file not provided or not found. Computing TF-IDF inline from directory...")
            tfidf_weights = compute_tfidf_weights(list(docs.values()))

    results = []

    
    if engine in ['minhash', 'both']:
        click.echo("Executing Path 1: Initializing MinHash structural sub-band buckets...")
        minhash_engine = MinHashEngine()
        lsh = LSH(threshold=threshold, signature_len=128)
        
        for doc_id, content in docs.items():
            tokens = tokenize(content)
            shingles = generate_shingles(tokens, k=shingle_size)
            signature = minhash_engine.compute_signature(shingles)
            lsh.add_document(doc_id, signature)
            
        minhash_pairs = lsh.get_candidate_pairs()
        for doc_a, doc_b, sim in minhash_pairs:
            results.append({"doc_a": doc_a, "doc_b": doc_b, "similarity": sim, "method": "MinHash+LSH"})

    
    if engine in ['simhash', 'both']:
        click.echo("Executing Path 2: Constructing SimHash 64-bit signatures...")
        simhash_engine = SimHashEngine(tfidf_weights=tfidf_weights)
        fingerprints = {}
        
        for doc_id, content in docs.items():
            tokens = tokenize(content)
            fingerprints[doc_id] = simhash_engine.compute_fingerprint(tokens)
            
        click.echo("Performing proximity scan over fingerprint sets...")
        doc_ids = list(docs.keys())
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                id_a, id_b = doc_ids[i], doc_ids[j]
                h_dist = simhash_engine.hamming_distance(fingerprints[id_a], fingerprints[id_b])
                raw_sim = 1.0 - h_dist
                calibrated_sim = max(0.0, (raw_sim - 0.5) * 2)
                if calibrated_sim >= threshold:
                                    results.append({
                                        "doc_a": id_a, 
                                        "doc_b": id_b, 
                                        "similarity": round(calibrated_sim, 4), 
                                        "method": "SimHash"
                                    })

    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        df.to_csv(output, index=False)
        click.echo(f"Found {len(results)} candidate matches, exported to {output}")
    else:
        click.echo("No duplicates met your threshold criteria.")
        pd.DataFrame(columns=["doc_a", "doc_b", "similarity", "method"]).to_csv(output, index=False)


@cli.command("pairs")
@click.option("--pairs", required=True, help="CSV file with text fields and target tags ")
@click.option("--threshold", type=float, default=0.7, help="Similarity boundary used for classification")
@click.option("--text-col-a", default="text_a", help="Column key for first text field ")
@click.option("--text-col-b", default="text_b", help="Column key for second text field ")
@click.option("--label-col", default="label", help="Column tracking ground truth binary labels ")
@click.option("--limit", type=int, default=None, help="Maximum number of rows to pull ")
@click.option("--output", required=True, help="Detailed predictions target path ")
def pairs(pairs, threshold, text_col_a, text_col_b, label_col, limit, output):
    """Run an empirical side-by-side verification benchmark over both engineering paths """
    df = pd.read_csv(pairs)
    if limit is not None:
        df = df.sample(limit, random_state=42)

    click.echo("Generating inline vocabulary index to balance SimHash execution...")
    all_texts = df[text_col_a].dropna().tolist() + df[text_col_b].dropna().tolist()
    tfidf_weights = compute_tfidf_weights(all_texts)

    minhash_engine = MinHashEngine()
    simhash_engine = SimHashEngine(tfidf_weights=tfidf_weights)

    
    stats = {
        "minhash": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "time": 0.0},
        "simhash": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "time": 0.0}
    }

    detailed_records = []

    for idx, row in df.iterrows():
        text_a, text_b = str(row[text_col_a]), str(row[text_col_b])
        true_label = int(row[label_col])

        tokens_a, tokens_b = tokenize(text_a), tokenize(text_b)
        shingles_a = generate_shingles(tokens_a, k=3)
        shingles_b = generate_shingles(tokens_b, k=3)

        # ---- BENCHMARK MINHASH ----
        t0 = time.perf_counter()
        sig_a = minhash_engine.compute_signature(shingles_a)
        sig_b = minhash_engine.compute_signature(shingles_b)
        m_sim = minhash_engine.estimate_similarity(sig_a, sig_b)
        stats["minhash"]["time"] += (time.perf_counter() - t0)
        
        m_pred = 1 if m_sim >= threshold else 0
        if true_label == 1:
            stats["minhash"]["tp" if m_pred == 1 else "fn"] += 1
        else:
            stats["minhash"]["fp" if m_pred == 1 else "tn"] += 1

        # ---- BENCHMARK SIMHASH ----
        t0 = time.perf_counter()
        fp_a = simhash_engine.compute_fingerprint(tokens_a)
        fp_b = simhash_engine.compute_fingerprint(tokens_b)
        s_sim = 1.0 - simhash_engine.hamming_distance(fp_a, fp_b)
        stats["simhash"]["time"] += (time.perf_counter() - t0)

        s_pred = 1 if s_sim >= threshold else 0
        if true_label == 1:
            stats["simhash"]["tp" if s_pred == 1 else "fn"] += 1
        else:
            stats["simhash"]["fp" if s_pred == 1 else "tn"] += 1

        detailed_records.append({
            "index": idx,
            "true_label": true_label,
            "minhash_similarity": round(m_sim, 4),
            "minhash_prediction": m_pred,
            "simhash_similarity": round(s_sim, 4),
            "simhash_prediction": s_pred
        })

    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.DataFrame(detailed_records).to_csv(output, index=False)

    
    summary_metrics = []
    for model_name, data_dict in stats.items():
        tp, fp, fn, tn = data_dict["tp"], data_dict["fp"], data_dict["fn"], data_dict["tn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        summary_metrics.append({
            "pipeline": model_name,
            "threshold": threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "latency_seconds": round(data_dict["time"], 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        })

    metrics_df = pd.DataFrame(summary_metrics)
    metrics_path = "outputs/metrics.csv" 
    metrics_df.to_csv(metrics_path, index=False)

    click.echo(f"\nEvaluation Complete. Results stored at: {output}")
    click.echo(f"Comprehensive benchmarking log written directly to: {metrics_path} ")
    print(metrics_df[['pipeline', 'precision', 'recall', 'f1_score', 'latency_seconds']].to_string(index=False))


if __name__ == "__main__":
    cli()
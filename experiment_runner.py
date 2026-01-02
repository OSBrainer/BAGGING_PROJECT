import argparse
import sys
import time
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from solcx import compile_source, install_solc, set_solc_version
from web3 import Web3


# Contract source reused from blockchain_run.py
CONTRACT_SOURCE = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract RandomForestDecision {
    struct Vote {
        uint treeId;
        int prediction;
        uint timestamp;
    }

    Vote[] public votes;
    event VoteRecu(uint treeId, int prediction);

    function submitVote(uint _treeId, int _prediction) public {
        votes.push(Vote(_treeId, _prediction, block.timestamp));
        emit VoteRecu(_treeId, _prediction);
    }

    function getConsensus() public view returns (string memory, int, int) {
        int count0 = 0;
        int count1 = 0;
        for(uint i=0; i<votes.length; i++){
            if(votes[i].prediction == 0) count0++;
            else count1++;
        }
        string memory verdict = (count1 > count0) ? "CLASSE 1" : "CLASSE 0";
        return (verdict, count0, count1);
    }
}
'''


def connect(ganache_url: str) -> Web3:
    w3 = Web3(Web3.HTTPProvider(ganache_url))
    if not w3.is_connected():
        raise RuntimeError(f"Cannot connect to Ganache at {ganache_url}")
    w3.eth.default_account = w3.eth.accounts[0]
    return w3


def deploy_contract(w3: Web3) -> Tuple[str, any]:
    install_solc('0.8.0')
    set_solc_version('0.8.0')
    compiled_sol = compile_source(CONTRACT_SOURCE, output_values=['abi', 'bin'], solc_version='0.8.0')
    _, contract_interface = compiled_sol.popitem()
    contract = w3.eth.contract(abi=contract_interface['abi'], bytecode=contract_interface['bin'])
    tx_hash = contract.constructor().transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    address = tx_receipt.contractAddress
    return address, w3.eth.contract(address=address, abi=contract_interface['abi'])


def load_model(path: str):
    return joblib.load(path)


def select_from_dataset(df: pd.DataFrame, label_col: str, force_class: Optional[int], mixed: bool) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Return X, y, and indices chosen from dataset with optional force/mixed logic."""
    labels = df[label_col].values
    X = df.drop(columns=[label_col]).values

    chosen_indices = []
    samples = []
    y_true = []

    if force_class is not None:
        idxs = np.where(labels == force_class)[0]
        if len(idxs) == 0:
            raise RuntimeError(f"No samples with class {force_class} in dataset.")
        chosen = np.random.choice(idxs)
        chosen_indices.append(int(chosen))
        samples.append(X[chosen])
        y_true.append(labels[chosen])
    elif mixed:
        # pick one fraud (1) and one valid (0) if available
        idx1 = np.where(labels == 1)[0]
        idx0 = np.where(labels == 0)[0]
        if len(idx1) == 0 or len(idx0) == 0:
            raise RuntimeError("Mixed requested but dataset lacks both classes.")
        for pool in (idx1, idx0):
            chosen = np.random.choice(pool)
            chosen_indices.append(int(chosen))
            samples.append(X[chosen])
            y_true.append(labels[chosen])
    else:
        chosen = np.random.choice(len(labels))
        chosen_indices.append(int(chosen))
        samples.append(X[chosen])
        y_true.append(labels[chosen])

    return np.vstack(samples), np.array(y_true), chosen_indices


def prepare_data(
    n_features: int,
    mixed: bool,
    feature_constant: Optional[float],
    feature_first: Optional[float],
    dataset: Optional[pd.DataFrame],
    label_col: Optional[str],
    force_class: Optional[int],
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
    if dataset is not None and label_col is not None:
        data, labels, idxs = select_from_dataset(dataset, label_col, force_class, mixed)
        return data, labels, idxs

    if mixed:
        low_val = 0.2 if feature_constant is None else feature_constant
        high_val = 0.8 if feature_constant is None else feature_constant
        low = np.ones((1, n_features)) * low_val
        high = np.ones((1, n_features)) * high_val
        if feature_first is not None:
            low[0, 0] = feature_first
            high[0, 0] = feature_first
        return np.vstack([low, high]), None, [0, 1]

    if feature_constant is not None:
        sample = np.ones((1, n_features)) * feature_constant
    else:
        sample = np.random.rand(1, n_features)
    if feature_first is not None:
        sample[0, 0] = feature_first
    return sample, None, [0]


def run_votes(
    w3: Web3,
    contract,
    arbres: List,
    data: np.ndarray,
    labels: Optional[np.ndarray],
    malicious_count: int,
    faulty_random_count: int,
    lazy_every_n: int,
    sybil_votes: int,
    sybil_class: int,
    limit_trees: int,
    correlated_failure: bool,
    bias_shift: float,
    bias_threshold: float,
    timing_delay: float,
    dominance_boost: bool,
    mask_top_k: int,
    force_class1_count: int,
) -> Tuple[List[int], int, int, str]:
    votes_log = []
    malicious_indices = set(range(malicious_count))
    faulty_indices = set(range(malicious_count, malicious_count + faulty_random_count))
    lazy_indices = set(range(0, limit_trees, lazy_every_n)) if lazy_every_n else set()

    # Optional ground-truth debugging
    if labels is not None and len(labels) > 0:
        sample_labels = labels
        print(f"🔍 Ground truth labels for selected samples: {sample_labels.tolist()}")

    submission_counter = 0

    for i, arbre in enumerate(arbres[:limit_trees]):
        sample_idx = i % data.shape[0]
        sample = data[sample_idx:sample_idx + 1].copy()

        # Correlated failure: mask top-k features to drive many trees down same path
        if correlated_failure:
            k = mask_top_k if mask_top_k > 0 else min(5, sample.shape[1])
            sample[0, :k] = 0.0

        # Soft voting path: use proba to allow bias shifting at the probability level
        proba = getattr(arbre, "predict_proba", None)
        if proba:
            prob = float(proba(sample)[0][1])
            prob = float(np.clip(prob + bias_shift, 0.0, 1.0))
            prediction = int(prob >= bias_threshold)
        else:
            prediction = int(arbre.predict(sample)[0])

        # For demonstrations, force early votes to class 1 to break all-zero runs
        if force_class1_count and i < force_class1_count:
            prediction = 1

        if i in malicious_indices:
            prediction = 1 - prediction
        if i in faulty_indices:
            prediction = np.random.randint(0, 2)
        if i in lazy_indices:
            print(f"   Node {i+1}/{limit_trees} skipped (lazy)")
            continue

        repeat = 2 if (dominance_boost and i == 0) else 1
        for r in range(repeat):
            votes_log.append(prediction)
            tree_id = submission_counter
            tx_hash = contract.functions.submitVote(tree_id, prediction).transact()
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            submission_counter += 1
            print(f"   Node {i+1}/{limit_trees} vote#{r+1}: {prediction} -> block {receipt.blockNumber} (Gas: {receipt.gasUsed})")

        if timing_delay > 0:
            time.sleep(timing_delay)

    # Optional Sybil flood: extra identities voting the same class
    for j in range(sybil_votes):
        tree_id = submission_counter
        tx_hash = contract.functions.submitVote(tree_id, sybil_class).transact()
        w3.eth.wait_for_transaction_receipt(tx_hash)
        votes_log.append(sybil_class)
        submission_counter += 1

    verdict, v0, v1 = contract.functions.getConsensus().call()
    return votes_log, v0, v1, verdict


def parse_args():
    parser = argparse.ArgumentParser(description="Run blockchain bagging scenarios for slides.")
    parser.add_argument("--ganache-url", default="http://127.0.0.1:7545")
    parser.add_argument("--model", default="classifier.pkl")
    parser.add_argument("--mixed-data", action="store_true", help="Force contrasting samples to provoke disagreements.")
    parser.add_argument("--malicious-count", type=int, default=0, help="Number of trees that flip their votes.")
    parser.add_argument("--faulty-random-count", type=int, default=0, help="Number of trees sending random votes.")
    parser.add_argument("--lazy-every-n", type=int, default=0, help="Skip every Nth tree (0 = disabled).")
    parser.add_argument("--sybil-votes", type=int, default=0, help="Number of extra Sybil votes to append.")
    parser.add_argument("--sybil-class", type=int, choices=[0, 1], default=1, help="Vote value for Sybil nodes.")
    parser.add_argument("--limit-trees", type=int, default=0, help="Limit number of trees used (0 = all).")
    parser.add_argument("--correlated-failure", action="store_true", help="Push an extreme feature value to induce correlated errors.")
    parser.add_argument("--bias-shift", type=float, default=0.0, help="Additive shift to predicted probability to expose systematic bias.")
    parser.add_argument("--bias-threshold", type=float, default=0.5, help="Decision threshold when using bias shift / soft voting.")
    parser.add_argument("--timing-delay", type=float, default=0.0, help="Sleep seconds between votes to simulate delayed voting.")
    parser.add_argument("--dominance-boost", action="store_true", help="Duplicate vote of first tree to show dominance effect.")
    parser.add_argument("--force-class1-count", type=int, default=0, help="Force the first N votes to class 1 (demo break-glass).")
    parser.add_argument("--feature-constant", type=float, default=None, help="Set all features to this constant value.")
    parser.add_argument("--feature-first", type=float, default=None, help="Set only the first feature to this value.")
    parser.add_argument("--data-csv", type=str, default=None, help="CSV file with features and label column.")
    parser.add_argument("--label-col", type=str, default=None, help="Label column name in CSV.")
    parser.add_argument("--force-class", type=int, choices=[0, 1], default=None, help="Force selecting a sample of this class from dataset.")
    parser.add_argument("--mask-top-k", type=int, default=0, help="Mask the first K features (set to 0) to induce correlated failure.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    print(f"🚀 Running scenario with model={args.model}")
    print(f"   mixed_data={args.mixed_data}, malicious={args.malicious_count}, faulty_random={args.faulty_random_count}, lazy_every_n={args.lazy_every_n}, sybil_votes={args.sybil_votes}, sybil_class={args.sybil_class}")
    print(f"   correlated_failure={args.correlated_failure}, mask_top_k={args.mask_top_k}, bias_shift={args.bias_shift}, bias_threshold={args.bias_threshold}, timing_delay={args.timing_delay}, dominance_boost={args.dominance_boost}")
    print(f"   force_class1_count={args.force_class1_count}, feature_constant={args.feature_constant}, feature_first={args.feature_first}, data_csv={args.data_csv}, label_col={args.label_col}, force_class={args.force_class}")

    try:
        w3 = connect(args.ganache_url)
        print(f"✅ Connected to Ganache at {args.ganache_url} | Block: {w3.eth.block_number}")
    except Exception as exc:
        print(f"❌ Connection error: {exc}")
        sys.exit(1)

    try:
        address, contract = deploy_contract(w3)
        print(f"✅ Contract deployed at {address}")
    except Exception as exc:
        print(f"❌ Contract deployment failed: {exc}")
        sys.exit(1)

    try:
        model = load_model(args.model)
        n_features = getattr(model, "n_features_in_", None) or 10
        arbres = getattr(model, "estimators_", None)
        if arbres is None:
            raise RuntimeError("Model has no estimators_ (not a RandomForest).")
        print(f"ℹ️ Model loaded with {len(arbres)} trees, expecting {n_features} features.")
    except Exception as exc:
        print(f"❌ Model error: {exc}")
        sys.exit(1)

    dataset = None
    if args.data_csv:
        try:
            dataset = pd.read_csv(args.data_csv)
        except Exception as exc:
            print(f"❌ Failed to read dataset {args.data_csv}: {exc}")
            sys.exit(1)

    data, labels, idxs = prepare_data(
        n_features=n_features,
        mixed=args.mixed_data,
        feature_constant=args.feature_constant,
        feature_first=args.feature_first,
        dataset=dataset,
        label_col=args.label_col,
        force_class=args.force_class,
    )
    if labels is not None:
        print(f"📌 Selected dataset indices: {idxs}")
        print(f"📌 Ground truth y_true: {labels.tolist()}")
        # Also show RF prediction on selected samples
        try:
            rf_probs = model.predict_proba(data)[:, 1]
            rf_preds = (rf_probs >= 0.5).astype(int)
            print(f"📌 RF preds: {rf_preds.tolist()}, RF probs: {rf_probs.round(3).tolist()}")
        except Exception:
            try:
                rf_preds = model.predict(data)
                print(f"📌 RF preds (no proba): {rf_preds.tolist()}")
            except Exception as exc:
                print(f"⚠️ Could not compute RF predictions on selected samples: {exc}")
    limit = args.limit_trees or len(arbres)

    t0 = time.time()
    votes_log, v0, v1, verdict = run_votes(
        w3=w3,
        contract=contract,
        arbres=arbres,
        data=data,
        labels=labels,
        malicious_count=args.malicious_count,
        faulty_random_count=args.faulty_random_count,
        lazy_every_n=args.lazy_every_n,
        sybil_votes=args.sybil_votes,
        sybil_class=args.sybil_class,
        limit_trees=limit,
        correlated_failure=args.correlated_failure,
        mask_top_k=args.mask_top_k,
        bias_shift=args.bias_shift,
        bias_threshold=args.bias_threshold,
        timing_delay=args.timing_delay,
        dominance_boost=args.dominance_boost,
        force_class1_count=args.force_class1_count,
    )
    elapsed = time.time() - t0

    print("\n⚖️  --- CONSENSUS ---")
    print(f"🗳️ Votes for 0: {v0}")
    print(f"🗳️ Votes for 1: {v1}")
    print(f"🏆 Final verdict: {verdict}")
    print(f"⏱️ Duration: {elapsed:.2f}s | Votes counted (including Sybil): {len(votes_log)}")
    print(f"🔎 Votes detail (first 30): {votes_log[:30]}")
    print("\nCopy these numbers into your slides for the scenario.")


if __name__ == "__main__":
    main()

"""
Author: David Kuter
Date: 2 December 2025

Generate Tanimoto Similarity distribution of the molecules in the training dataset and
compare these with compounds in the commercial dataset to see if they are out-of-distribution
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import combinations
from joblib import Parallel, delayed
from loguru import logger
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from tqdm import tqdm


def df_to_rdkit_bitvectors(
    df: pd.DataFrame, n_jobs: int = -1
) -> dict[int, ExplicitBitVect]:
    """
    Convert a DataFrame of 0/1 fingerprint columns into RDKit ExplicitBitVect objects
    using joblib multiprocessing for parallelization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each row is a fingerprint and fingerprint bits are stored
        in numeric (0/1) columns.
    n_jobs : int, optional
        Number of jobs for parallel processing. -1 uses all available cores.
    """

    nbits = len(df.columns)

    def _convert_row(row_data):
        """Convert a single row to an ExplicitBitVect."""
        idx, row = row_data
        bv = ExplicitBitVect(nbits)
        for i, bit in enumerate(row):
            if int(bit):
                bv.SetBit(i)
        return idx, bv

    # Create list of (index, row) tuples
    rows = list((idx, row) for idx, row in df.iterrows())

    # Convert rows in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_convert_row)(row_data) for row_data in rows
    )

    # Convert results to dictionary
    bvs: dict[int, ExplicitBitVect] = {idx: bv for idx, bv in results}

    return bvs


def load_dataset(data_path, purchase_col: str = "Purchase", n_cpu: int = -1) -> tuple[dict[int, ExplicitBitVect], list]:
    # Load the dataset
    df = pd.read_csv(data_path, index_col="Name")

    # Remove Purchased column if present and get purchased indices. This is only present in the commercial dataset
    purchased = []
    if purchase_col in df.columns:
        purchased = df[df[purchase_col] == 1].index.tolist()
        df = df.drop(columns=[purchase_col])

    # Drop rows with NaN values
    all_rows = df.shape[0]
    df = df.dropna()
    dropped_rows = all_rows - df.shape[0]
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows with NaN values. {df.shape[0]} rows remain.")

    # Convert DataFrames to RDKit ExplicitBitVect dictionaries
    bv_dict = df_to_rdkit_bitvectors(df, n_jobs=n_cpu)

    return bv_dict, purchased


def _compute_similarity(bv1, bv2, idx1, idx2) -> tuple[tuple[int, int], float]:
    """Compute Tanimoto similarity between two bitvectors. Indexes are included for tracking."""
    return (idx1, idx2), DataStructs.TanimotoSimilarity(bv1, bv2)


def compute_pairwise_tanimoto(
    bv_dict: dict[int, ExplicitBitVect], n_jobs: int = -1
) -> dict[tuple[int, int], float]:
    """
    Compute pairwise Tanimoto similarities for all bitvectors in the provided dictionary
    using joblib multiprocessing for parallelization.

    Parameters
    ----------
    bv_dict : dict[int, ExplicitBitVect]
        Dictionary mapping IDs to RDKit ExplicitBitVect objects.
    n_jobs : int, optional
        Number of jobs for parallel processing. -1 uses all available cores.

    Returns
    -------
    dict[tuple[int, int], float]
        Dictionary mapping pairs of IDs to their Tanimoto similarity.
    """
    # Generate all pairs
    pairs = list(combinations(bv_dict.keys(), 2))

    # Compute similarities in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_similarity)(
            bv_dict[combo[0]], bv_dict[combo[1]], combo[0], combo[1]
        )
        for combo in pairs
    )

    # Convert results to dictionary
    similarity_dict = {pair: sim for pair, sim in results}

    return similarity_dict


def compute_groupwise_tanimoto(
    commercial_bv_dict: dict[int, ExplicitBitVect],
    training_bv_dict: dict[int, ExplicitBitVect],
    n_jobs: int = -1,
) -> dict[tuple[int, int], float]:
    # Generate all pairs between commercial and training sets
    pairs = [
        (commercial_idx, training_idx)
        for training_idx in training_bv_dict.keys()
        for commercial_idx in commercial_bv_dict.keys()
    ]

    # Compute similarities in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_similarity)(
            commercial_bv_dict[combo[0]], training_bv_dict[combo[1]], combo[0], combo[1]
        )
        for combo in pairs
    )

    # Convert results to dictionary
    similarity_dict = {pair: sim for pair, sim in results}

    return similarity_dict


def plot_histogram(data: pd.Series, title: str, xlabel: str, ylabel: str, output_path: str, bins: int = 20):

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(output_path)
    plt.close()

def get_training_similarities(
    training_dataset_path: str, output_dir: str, n_cpu: int = -1
) -> pd.DataFrame:
    training_sim_path = os.path.join(output_dir, "training_tanimoto_similarities.csv")

    # Check if training similarities file exists
    if os.path.exists(training_sim_path) is False:
        # Load training dataset
        training_dict, _ = load_dataset(training_dataset_path, n_cpu=n_cpu)

        # Determine Distribution of Tanimoto Similarities of Training Dataset
        training_similarity_distribution = compute_pairwise_tanimoto(
            training_dict, n_jobs=n_cpu
        )

        # Save training similarities to CSV
        df_training_sim = pd.DataFrame(
            [
                (k1, k2, v)
                for (k1, k2), v in training_similarity_distribution.items()
                if k1 != k2
            ],
            columns=["Compound1", "Compound2", "Tanimoto_Similarity"],
        )
        df_training_sim.to_csv(training_sim_path, index=False)

    # Load existing training similarities
    else:
        logger.info(f"Loading existing training similarities from {training_sim_path}")
        df_training_sim = pd.read_csv(training_sim_path)

    return df_training_sim


def get_commercial_similarities(
    commercial_dataset_path: str, output_dir: str, n_cpu: int = -1
) -> pd.DataFrame:
    commercial_sim_path = os.path.join(
        output_dir, "commercial_tanimoto_similarities.csv"
    )

    # Check if commercial similarities file exists
    if os.path.exists(commercial_sim_path) is False:
        # Load commercial dataset
        commercial_dict, purchased = load_dataset(commercial_dataset_path, n_cpu=n_cpu)

        # Load training dataset
        training_dict, _ = load_dataset(training_dataset_path, n_cpu=n_cpu)

        # Determine Distribution of Tanimoto Similarities of Commercial Dataset
        commercial_similarity_distribution = compute_groupwise_tanimoto(
            commercial_dict, training_dict, n_jobs=n_cpu
        )

        # Save commercial similarities to CSV
        df_commercial_sim = pd.DataFrame(
            [
                (k1, k2, v)
                for (k1, k2), v in commercial_similarity_distribution.items()
                if k1 != k2
            ],
            columns=["Commercial_Compound", "Training_Compound", "Tanimoto_Similarity"],
        )
        df_commercial_sim["Purchase"] = df_commercial_sim["Commercial_Compound"].apply(
            lambda x: 1 if x in purchased else 0
        )
        df_commercial_sim.to_csv(commercial_sim_path, index=False)

    # Load existing commercial similarities
    else:
        logger.info(f"Loading existing commercial similarities from {commercial_sim_path}")
        df_commercial_sim = pd.read_csv(commercial_sim_path)
    return df_commercial_sim


def get_random_similarities(
    random_dataset_path: str, output_dir: str, n_cpu: int = -1, n_boots: int = 100, seed: int = 42
) -> pd.DataFrame:
    random_sim_path = os.path.join(output_dir, "random_tanimoto_similarities.csv")

    # Check if random similarities file exists
    if os.path.exists(random_sim_path) is False:
        random_state = np.random.RandomState(seed)
        extension = random_dataset_path.split(".")[-1]
        # Load random dataset and fingerprints
        df_random = pd.read_csv(random_dataset_path, names=["ID1", "SMILES1", "ID2", "SMILES2"], sep="\t")
        random_dict, _ = load_dataset(random_dataset_path.replace(f".{extension}", ".FP.csv"), n_cpu=n_cpu)

        # Drop rows with missing fingerprints
        all_rows = df_random.shape[0]
        df_random = df_random[df_random["ID1"].isin(random_dict.keys()) & df_random["ID2"].isin(random_dict.keys())]
        dropped_rows = all_rows - df_random.shape[0]
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing fingerprints. {df_random.shape[0]} rows remain.")

        # Compute similarities by bootstrapping and shuffling the second compound column
        random_similarities = dict()
        for i in tqdm(range(n_boots)):
            temp = df_random.copy()
            pairs = zip(temp["ID1"].tolist(), temp["ID2"].sample(frac=1, random_state=random_state).tolist())
            results = Parallel(n_jobs=n_cpu)(
                delayed(_compute_similarity)(
                    random_dict[combo[0]], random_dict[combo[1]], combo[0], combo[1]
                )
                for combo in pairs
            )
            # Just store similarities
            random_similarities[i] = [sim for _, sim in results]

        # Save random similarities to CSV
        df_random_sim = pd.DataFrame.from_dict(random_similarities, orient='index', columns=[f"tani_{i}" for i in range(len(random_similarities[0]))])
        df_random_sim.index.name = "Iteration"
        df_random_sim.to_csv(random_sim_path, index=True)
    else:
        logger.info(f"Loading existing random similarities from {random_sim_path}")
        df_random_sim = pd.read_csv(random_sim_path, index_col="Iteration")

    return df_random_sim


def compute_quartiles(df:pd.DataFrame, quartiles: list[float]) -> pd.DataFrame:
    # Compute quartiles across bootstrapped iterations
    df = df.quantile(quartiles)

    # Get the mean of each quartile across iterations
    df_stats = df.mean(axis=1).to_frame(name="Mean_Tanimoto_Similarity")
    df_stats.index.name = "Percentile"
    df_stats["Standard_Deviation"] = df.std(axis=1).values
    return df_stats



def get_random_quartiles(random_dataset_path: str, quartiles: list[float], output_dir: str, n_cpu: int = -1):


    # Compute quartiles across bootstrapped iterations
    df_random_quartiles = df_random_sim.T.quantile(quartiles)

    # Get the mean of each quartile across iterations
    df_stats = df_random_quartiles.mean(axis=1).to_frame(name="Mean_Tanimoto_Similarity")
    df_stats.index.name = "Percentile"
    df_stats["Standard_Deviation"] = df_random_quartiles.std(axis=1).values

    print("Random Dataset Tanimoto Similarity Quartiles:")
    print(df_stats)
    return df_stats


def num_commercial_in_quartile(
    df_max_commercial_sim: pd.DataFrame,
    df_quartiles: pd.DataFrame,
    quartiles: list[float],
):
    # Determine number of commercial compounds above quartiles
    stats = []
    for q in quartiles:
        threshold = df_quartiles.loc[q, "Mean_Tanimoto_Similarity"]
        n_above = df_max_commercial_sim[df_max_commercial_sim["Tanimoto_Similarity"] >= threshold].shape[0]
        n_purchased_above = df_max_commercial_sim[
            (df_max_commercial_sim["Tanimoto_Similarity"] >= threshold) &
            (df_max_commercial_sim["Purchase"] == 1)
        ].shape[0]
        stats.append((q, threshold, n_above, n_purchased_above))
    df_stats = pd.DataFrame(
        stats,
        columns=[
            "Percentile",
            "Tanimoto_Threshold",
            "Num_Commercial_Above_Threshold",
            "Num_Purchased_Above_Threshold",
        ],
    )
    print(f"Total Commercial Compounds: {df_max_commercial_sim.shape[0]}")
    print(f"Total Purchased Commercial Compounds: {df_max_commercial_sim[df_max_commercial_sim['Purchase'] == 1].shape[0]}")
    print(df_stats)
    return df_stats

def main(
    training_dataset_path: str,
    commercial_dataset_path: str,
    random_dataset_path: str,
    output_dir: str,
    n_cpu: int = -1,
):
    # Create output files
    os.makedirs(output_dir, exist_ok=True)

    # Get training similarities (will create file if not present)
    df_training_sim = get_training_similarities(
        training_dataset_path, output_dir, n_cpu=n_cpu
    )
    df_max_training_sim = df_training_sim.loc[
        df_training_sim.groupby("Compound1")["Tanimoto_Similarity"].idxmax()
    ]

    # Get commercial similarities (will create file if not present)
    df_commercial_sim = get_commercial_similarities(
        commercial_dataset_path, output_dir, n_cpu=n_cpu
    )

    # Only consider the maximum similarity for each commercial compound
    df_max_commercial_sim = df_commercial_sim.loc[
        df_commercial_sim.groupby("Commercial_Compound")["Tanimoto_Similarity"].idxmax()
    ]

    # Get random similarities
    df_random_sim = get_random_similarities(
        random_dataset_path, output_dir, n_cpu=n_cpu
    )

    # Get random quartiles (will create file if not present)
    quartiles = [0.5, 0.75, 0.8, 0.85, 0.88, 0.9, 0.95, 0.99]

    print("Random Dataset Tanimoto Similarity Quartiles:")
    df_random_quartiles = compute_quartiles(
        df_random_sim.T, quartiles
    )
    print(df_random_quartiles)

    print("Training Dataset Tanimoto Similarity Quartiles:")
    print("Mean and Std:", df_training_sim["Tanimoto_Similarity"].mean(), " +/- ",df_training_sim["Tanimoto_Similarity"].std())
    df_training_quartiles =  df_training_sim["Tanimoto_Similarity"].quantile(quartiles).to_frame(name="Mean_Tanimoto_Similarity")
    df_training_quartiles.index.name = "Percentile"
    print(df_training_quartiles)

    print("Training Dataset of Maximum Similarities Quartiles:")
    print("Mean and Std:", df_max_training_sim["Tanimoto_Similarity"].mean(), " +/- ",df_max_training_sim["Tanimoto_Similarity"].std())
    min_quartiles = [1 - q for q in quartiles]
    df_training_max_quartiles =  df_max_training_sim["Tanimoto_Similarity"].quantile(min_quartiles).to_frame(name="Mean_Tanimoto_Similarity")
    df_training_max_quartiles.index.name = "Percentile"
    print(df_training_max_quartiles)

    # Summary
    print("Summary of Commercial Compounds in Relation to Random Dataset Quartiles:")
    num_commercial_in_quartile(
        df_max_commercial_sim, df_random_quartiles, quartiles
    )
    print("Summary of Commercial Compounds in Relation to Training Dataset Quartiles:")
    num_commercial_in_quartile(
        df_max_commercial_sim, df_training_quartiles, quartiles
    )
    print("Summary of Commercial Compounds in Relation to Training Dataset of Maximum Similarities Quartiles:")
    num_commercial_in_quartile(
        df_max_commercial_sim, df_training_max_quartiles, min_quartiles
    )

    print(df_max_commercial_sim[df_max_commercial_sim["Purchase"] == 1].sort_values(by="Tanimoto_Similarity", ascending=True))

if __name__ == "__main__":
    # Required inputs
    n_cpu = 6
    output_dir = "output_data"
    training_dataset_path = "./input_data/Training_Set_FP.csv"
    commercial_dataset_path = "./input_data/Commercial_Data.csv"
    random_dataset_path = "./input_data/chembl30_50K.mfp0.pairs.txt"
    main(
        training_dataset_path,
        commercial_dataset_path,
        random_dataset_path=random_dataset_path,
        output_dir=output_dir,
        n_cpu=n_cpu,
    )

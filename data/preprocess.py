import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

import argparse
from args import get_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)

    pairs = pd.read_csv(args.pair_data_path, sep="\t", index_col=args.pair_id_column_name)  # TODO
    question_embeddings = pd.read_csv(args.question_data_path, sep="\t", index_col=args.question_id_column_name)
    # passage_embeddings = pd.read_csv(args.passage_data_path, sep="\t", index_col=args.passage_id_column_name)

    pairs = pairs.groupby("question").agg({
        "passage_index": list,
        "label": list,
    }).reset_index()
    question_id = pairs["question"].values
    question_embedding = question_embeddings.values
    passage_ids = pairs["passage_index"].values
    # passage_embeddings = np.array([passage_embeddings[passage_embeddings.index.isin(passage_id_list)].values for passage_id_list in passage_ids])
    labels = pairs["label"].values
    data = {
        "question_id": question_id,
        "question_embedding": question_embedding,
        "passage_ids": passage_ids,
        # "passage_embeddings": passage_embeddings,
        "labels": labels,
    }
    data = Dataset.from_dict(data)
    data.save_to_disk(args.output_dir)

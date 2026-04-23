import os
import itertools
import traceback
import gc
import torch

from train.train import ex
from sacred.observers import FileStorageObserver

param_grids = [
    {
        "loss": ["arcface"],
        "margin": [28.6, 45, 90],
        "embedding_dim": [128],
        "cluster_algorithm": ["kmeans", "agglomerative"],
        "n_prototypes_per_class": [20],
        "epochs": [50],
    }
]


def generate_experiments(grid):
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def main():
    for param_grid in param_grids:
        experiments = generate_experiments(param_grid)
        total_exps = len(experiments)

        for idx, config in enumerate(experiments, 1):
            print(f"\n[{idx}/{total_exps}] Starting Experiment...")

            experiment_folder_name = f"ddi_{config['loss']}_m{config['margin']}_dim{config['embedding_dim']}_p{config['n_prototypes_per_class']}_cluster{config['cluster_algorithm']}"
            save_path = os.path.join("results/ours", experiment_folder_name)

            ex.observers = [FileStorageObserver(save_path)]

            try:
                ex.run(config_updates=config)
                print(f"\n[{idx}/{total_exps}] Finished successfully.")

            except Exception as e:
                print(f"Experiment {idx} failed.")
                traceback.print_exc()

            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

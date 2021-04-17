import movielens
import torch.utils.data
import pandas as pd
from torch.multiprocessing import Pool, cpu_count, set_start_method
import optimizer_training_experiment
import proxpt_training_experiment
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def run(runner):
    return runner.run()


if __name__ == "__main__":
    set_start_method('spawn')

    W_train, y_train = movielens.load()
    ds = torch.utils.data.TensorDataset(W_train, y_train)
    ds = torch.utils.data.Subset(ds, range(1000))

    num_features = W_train.shape[1]
    max_nnz = W_train.sum(dim=1).max().item()
    max_proxpt_lr = 1. / (2*max_nnz + 1)

    embedding_dim = 20
    attempts = 10
    epochs = 10
    optimizer_lrs = np.geomspace(1e-5, 1e-1, 10)
    proxpt_lrs = np.geomspace(1e-5, max_proxpt_lr, 10)

    def make_optimizer_runners(optimizer_class):
        return optimizer_training_experiment.make_runners(
            ds=ds,
            optimizer_class=optimizer_class,
            lrs=optimizer_lrs,
            attempts=attempts,
            epochs=epochs,
            num_features=num_features,
            embedding_dim=embedding_dim
        )
    optimizer_runners = [
        runner
        for optimizer_class in [torch.optim.SGD, torch.optim.Adam, torch.optim.Adagrad]
        for runner in make_optimizer_runners(optimizer_class)
    ]
    proxpt_runners = proxpt_training_experiment.make_runners(
        ds=ds,
        lrs=proxpt_lrs,
        attempts=attempts,
        epochs=epochs,
        num_features=num_features,
        num_embeddings=embedding_dim
    )

    runners = [*optimizer_runners, *proxpt_runners]

    print(f'Using pool with {cpu_count()} workers to run {len(runners)} experiments')
    with Pool(cpu_count()) as pool:
        print('Pool started')
        result_list = pool.map(run, runners)
        results = pd.concat(result_list)
        results.to_csv('results.csv')
        print(results)

    sns.set()
    results = pd.read_csv('results.csv')
    best_loss = results \
        .groupby(['name', 'lr', 'attempt'], as_index=False) \
        .min()

    ax = sns.lineplot(x='lr', y='loss', hue='name', data=best_loss, err_style='band')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()

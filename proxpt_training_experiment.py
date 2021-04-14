import torch
import torch.utils.data
import torch.nn.functional as F
from fm_proxpt import ProxPtFMTrainer
from fm import FM
import pandas as pd


class Runner:
    def __init__(self, ds, lr, attempt, epochs, num_features, embedding_dim):
        self.ds = ds
        self.lr = lr
        self.attempt = attempt
        self.epochs = epochs
        self.num_features = num_features
        self.embedding_dim = embedding_dim

    def run(self):
        print(f'Starting: optimizer=ProximalPoint, lr={self.lr:.4}, attempt={self.attempt}')

        report = pd.DataFrame(columns=['name', 'lr', 'epoch', 'attempt', 'loss'])
        fm = FM(self.num_features, self.embedding_dim)
        trainer = ProxPtFMTrainer(fm, self.lr)

        for epoch in range(self.epochs):
            epoch_loss = 0.
            for w_sample, y_sample in torch.utils.data.DataLoader(self.ds, shuffle=True, batch_size=1):
                (ignore, w_nz) = torch.nonzero(w_sample, as_tuple=True)
                y = y_sample.squeeze(1)

                with torch.no_grad():
                    pred = fm.forward(w_nz.unsqueeze(0))
                    loss = F.binary_cross_entropy_with_logits(pred, y)
                    epoch_loss += loss.item()

                    y_hat = (2 * y.item() - 1)  # transform 0/1 labels into -1/1
                    trainer.step(w_nz, y_hat)

            epoch_loss /= len(self.ds)
            report = report.append(pd.DataFrame.from_dict(
                {'name': 'ProximalPoint',
                 'loss': [epoch_loss],
                 'epoch': [epoch],
                 'lr': [self.lr],
                 'attempt': [self.attempt]}), sort=True)

        print(f'Done: optimizer=ProximalPoint, lr={self.lr:.4}, attempt={self.attempt}')
        return report


def make_runners(ds, lrs, attempts, epochs, num_features, num_embeddings):
    return [
        Runner(ds, lr, attempt, epochs, num_features, num_embeddings)
        for lr in lrs
        for attempt in range(attempts)
    ]

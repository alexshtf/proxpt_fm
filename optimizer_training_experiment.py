import torch
import torch.utils.data
import torch.nn.functional as F
from fm import FM
import pandas as pd


class Runner:
    def __init__(self, ds, optimizer_class, lr, attempt, epochs, num_features, embedding_dim):
        self.ds = ds
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.attempt = attempt
        self.epochs = epochs
        self.num_features = num_features
        self.embedding_dim = embedding_dim

    def run(self):
        print(f'Starting: optimizer={self.optimizer_class.__name__}, lr={self.lr:.4}, attempt={self.attempt}')
        report = pd.DataFrame(columns=['name', 'lr', 'epoch', 'attempt', 'loss'])
        fm = FM(self.num_features, self.embedding_dim)
        optimizer = self.optimizer_class(fm.parameters(), self.lr)

        for epoch in range(self.epochs):
            epoch_loss = 0.
            for w_sample, y_sample in torch.utils.data.DataLoader(self.ds, shuffle=True, batch_size=1):
                (ignore, w_nz) = torch.nonzero(w_sample, as_tuple=True)
                y = y_sample.squeeze(1)

                optimizer.zero_grad()
                pred = fm.forward(w_nz.unsqueeze(0))
                loss = F.binary_cross_entropy_with_logits(pred, y)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    epoch_loss += loss.item()

            epoch_loss /= len(self.ds)
            row = pd.DataFrame.from_dict(
                {'name': self.optimizer_class.__name__,
                 'loss': [epoch_loss],
                 'epoch': [epoch],
                 'lr': [self.lr],
                 'attempt': [self.attempt]})
            report = report.append(row, sort=True)

        print(f'Done: optimizer={self.optimizer_class.__name__}, lr={self.lr:.4}, attempt={self.attempt}')
        return report


def make_runners(ds, optimizer_class, lrs, attempts, epochs, num_features, embedding_dim):
    return [
        Runner(ds, optimizer_class, lr, attempt, epochs, num_features, embedding_dim)
        for lr in lrs
        for attempt in range(attempts)
    ]

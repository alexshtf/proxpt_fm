import torch
import torch.nn.functional as F
from fm import FM
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

import movielens
from fm_proxpt import ProxPtFMTrainer

W_train, y_train = movielens.load()

num_features = W_train.shape[1]
max_nnz = W_train.sum(dim=1).max().item()
step_size = 1. / (2*max_nnz + 1)
embedding_dim = 20
print(f'Training with step_size={step_size:.4} computed using max_nnz = {max_nnz}')

fm = FM(num_features, embedding_dim)

dataset = TensorDataset(W_train, y_train)
epochs_num = 10
# optimizer = torch.optim.Adam(fm.parameters(), lr=3e-4)
trainer = ProxPtFMTrainer(fm, step_size)
for epoch in range(epochs_num):
    sum_epoch_loss = 0.
    sum_pred = 0.
    sum_label = 0.
    desc = f'Epoch = {epoch}, loss = 0, pred = 0, label = 0, bias = 0'
    with tqdm(DataLoader(dataset, batch_size=1, shuffle=True), desc=desc) as pbar:
        def update_progress(idx):
            avg_epoch_loss = sum_epoch_loss / (idx + 1)
            avg_pred = sum_pred / (idx + 1)
            avg_label = sum_label / (idx + 1)
            desc = f'Epoch = {epoch}, loss = {avg_epoch_loss:.4}, pred = {avg_pred:.4}, ' \
                   f'label = {avg_label:.4}, bias = {fm.bias.item():.4}'
            pbar.set_description(desc)

        for i, (x_sample, y_sample) in enumerate(pbar):
            (ignore, w_nz) = torch.nonzero(x_sample, as_tuple=True)
            y = y_sample.squeeze(1)

            # optimizer.zero_grad()
            # pred = fm.forward(w_nz.unsqueeze(0))
            # loss = F.binary_cross_entropy_with_logits(pred, y)
            # loss.backward()
            # optimizer.step()
            #
            # with torch.no_grad():
            #     sum_epoch_loss += loss.item()
            #     sum_pred += torch.sigmoid(pred).item()
            #     sum_label += y.item()

            with torch.no_grad():
                pred = fm.forward(w_nz.unsqueeze(0))
                loss = F.binary_cross_entropy_with_logits(pred, y)
                sum_epoch_loss += loss.item()
                sum_pred += torch.sigmoid(pred).item()
                sum_label += y.item()

                y_hat = (2 * y.item() - 1)  # transform 0/1 labels into -1/1
                trainer.step(w_nz, y_hat)

            if (i > 0) and (i % 2000 == 0):
                update_progress(i)

        update_progress(i)

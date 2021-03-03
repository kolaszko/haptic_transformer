import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import HapticDataset
from models import HAPTR

params = {
    'num_classes': 8,
    'projection_dim': 16,
    'hidden_dim': 64,
    'nheads': 4,
    'num_encoder_layers': 4,
    'mlp_head_dim_1': 2048,
    'mlp_head_dim_2': 1024,
    'dropout': 0.5,
    'lr': 6e-4,
    'gamma': 0.995,
    'weight_decay': 1e-2
}


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_ds = HapticDataset(args.dataset_path, 'train_ds', signal_start=90, signal_length=params['hidden_dim'])
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_ds = HapticDataset(args.dataset_path, 'val_ds', signal_start=90, signal_length=params['hidden_dim'])
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)

    mlp_head_dims = (params['mlp_head_dim_1'], params['mlp_head_dim_2'])

    model = HAPTR(params['num_classes'], params['projection_dim'],
                  params['hidden_dim'], params['nheads'],
                  params['num_encoder_layers'], mlp_head_dims,
                  params['dropout'])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params['gamma'])

    train_step = 0
    val_step = 0
    best_acc = 0

    with SummaryWriter() as writer:

        for epoch in range(args.epochs):
            print(f'Epoch: {epoch}')

            # Training
            running_loss = 0.0
            mean_loss = 0.0
            correct = 0
            model.train(True)
            for step, data in enumerate(train_dataloader):
                s, labels = data[0].to(device), data[1].to(device)

                out = model(s.unsqueeze(1))
                optimizer.zero_grad()

                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(out.data, 1)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()
                mean_loss += loss.item()

                if step % 10 == 9:
                    writer.add_scalar('running_loss/train', running_loss / 10, train_step)
                    print(f'Running loss training: {running_loss / 10} in step: {train_step}')
                    running_loss = 0.0
                    train_step += 1

            writer.add_scalar('loss/train', mean_loss / len(train_ds), epoch)
            writer.add_scalar('accuracy/train', (100 * correct / len(train_ds)), epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            scheduler.step()

            # Validation
            running_loss = 0.0
            mean_loss = 0.0
            correct = 0

            model.train(False)
            with torch.no_grad():
                for step, data in enumerate(val_dataloader):
                    s, labels = data[0].to(device), data[1].to(device)

                    out = model(s.unsqueeze(1))
                    loss = criterion(out, labels)

                    _, predicted = torch.max(out.data, 1)
                    correct += (predicted == labels).sum().item()

                    running_loss += loss.item()
                    mean_loss += loss.item()
                    if step % 10 == 9:
                        writer.add_scalar('running_loss/val', running_loss / 10, val_step)
                        print(f'Running loss validation: {running_loss / 10} in step: {val_step}')
                        running_loss = 0.0
                        val_step += 1

            acc = (100 * correct / len(val_ds))
            if acc > best_acc:
                best_acc = acc

            writer.add_scalar('loss/val', mean_loss / len(val_ds), epoch)
            writer.add_scalar('accuracy/val', acc, epoch)

        writer.add_hparams(params, {'max_accuracy': best_acc})
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=512)

    args, _ = parser.parse_known_args()
    main(args)

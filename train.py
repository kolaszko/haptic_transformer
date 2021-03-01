import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse

from data import HapticDataset
from models import HAPTR, PositionalEncoding, SignalEncoder


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_ds = HapticDataset(args.dataset_path, 'train_ds')
    train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True)

    val_ds = HapticDataset(args.dataset_path, 'val_ds')
    val_dataloader = DataLoader(val_ds, batch_size=16, shuffle=True)

    model = HAPTR(8, 78, 6, 6)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 5e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter()
    train_step = 0
    val_step = 0

    for epoch in range(args.epochs):
        print(epoch)

        # Training
        running_loss = 0.0
        mean_loss = 0.0
        correct = 0

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

            if step % 100 == 99:
                writer.add_scalar('running_loss/train', running_loss / 100, train_step)
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
        with torch.no_grad():
            for step, data in enumerate(val_dataloader):
                s, labels = data[0].to(device), data[1].to(device)

                out = model(s.unsqueeze(1))
                loss = criterion(out, labels)

                _, predicted = torch.max(out.data, 1)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()
                mean_loss += loss.item()
                if step % 100 == 99:
                    writer.add_scalar('running_loss/val', running_loss / 100, val_step)
                    running_loss = 0.0
                    val_step += 1

        writer.add_scalar('loss/val', mean_loss / len(val_ds), epoch)
        writer.add_scalar('accuracy/val', (100 * correct / len(val_ds)), epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)

    args, _ = parser.parse_known_args()
    main(args)

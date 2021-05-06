import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import HapticDataset
from models import HAPTR
from utils import validate_and_save


def main(args):
    params = {
        'num_classes': args.num_classes,
        'projection_dim': args.projection_dim,
        'hidden_dim': args.hidden_dim,
        'nheads': args.nheads,
        'num_encoder_layers': args.num_encoder_layers,
        'feed_forward': args.feed_forward,
        'dropout': args.dropout,
        'lr': args.lr,
        'gamma': args.gamma,
        'weight_decay': args.weight_decay,
        'batch_size' : args.batch_size
    }

    print(params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_ds = HapticDataset(args.dataset_path, 'train_ds', signal_start=0, signal_length=params['hidden_dim'])

    train_dataloader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)

    val_ds = HapticDataset(args.dataset_path, 'val_ds', signal_start=0, signal_length=params['hidden_dim'])
    val_dataloader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=True)

    model = HAPTR(params['num_classes'], params['projection_dim'],
                  params['projection_dim'], params['nheads'],
                  params['num_encoder_layers'], params['feed_forward'],
                  params['dropout'])
    model.to(device)

    weight = torch.Tensor(train_ds.weights)
    w = weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)

    best_acc = 0

    with SummaryWriter() as writer:
        print('======== LOG ========')
        print(writer.log_dir)
        print('========    ========')

        for epoch in range(args.epochs):
            print(f'Epoch: {epoch}')

            # Training
            mean_loss = 0.0
            correct = 0

            model.train(True)
            for step, data in enumerate(train_dataloader):
                s, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                out = model(s.unsqueeze(1))
                loss = criterion(out, labels)
                loss.backward()

                optimizer.step()

                _, predicted = torch.max(out.data, 1)
                correct += (predicted == labels).sum().item()

                mean_loss += loss.item()

                print(f'Running loss training: {loss.item()} in step: {step}')

            writer.add_scalar('loss/train', mean_loss / len(train_ds), epoch)
            writer.add_scalar('accuracy/train', (100 * correct / len(train_ds)), epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            scheduler.step()

            # Validation
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

                    mean_loss += loss.item()

                    print(f'Running loss validation: {loss.item()} in step: {step}')

            acc = (100 * correct / len(val_ds))
            if acc > best_acc:
                best_acc = acc
                torch.save(model, os.path.join(writer.log_dir, 'model'))
                print('========== ACC ==========')
                print(best_acc)
                print(f'Epoch: {epoch}')
                print('========== === ==========')

            writer.add_scalar('loss/val', mean_loss / len(val_ds), epoch)
            writer.add_scalar('accuracy/val', acc, epoch)

        writer.flush()


    validate_and_save(model, val_dataloader, writer.log_dir, device, (params['hidden_dim'], 6))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--projection-dim', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=160)
    parser.add_argument('--nheads', type=int, default=4)
    parser.add_argument('--num-encoder-layers', type=int, default=2)
    parser.add_argument('--feed-forward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=1e-3)

    args, _ = parser.parse_known_args()
    main(args)


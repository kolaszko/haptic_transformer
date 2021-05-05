import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import HapticDataset
from models import HAPTR
from utils import validate_and_save

params = {
    'num_classes': 8,
    'projection_dim': 16,
    'hidden_dim': 160,
    'nheads': 4,
    'num_encoder_layers': 2,
    'dropout': 0.5,
    'lr': 1e-4,
    'gamma': 0.999,
    'weight_decay': 1e-4,
    'comment': ''
}

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_ds = HapticDataset(args.dataset_path, 'train_ds', signal_start=0, signal_length=params['hidden_dim'])

    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_ds = HapticDataset(args.dataset_path, 'val_ds', signal_start=0, signal_length=params['hidden_dim'])
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)

    model = HAPTR(params['num_classes'], params['projection_dim'],
                  params['hidden_dim'], params['nheads'],
                  params['num_encoder_layers'],
                  params['dropout'])
    model.to(device)

    weight = torch.Tensor(train_ds.weights)
    w = weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * 2, eta_min=5e-6)

    best_acc = 0

    with SummaryWriter() as writer:

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
                nn.utils.clip_grad_norm(model.parameters(), 1.0)
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

            writer.add_scalar('loss/val', mean_loss / len(val_ds), epoch)
            writer.add_scalar('accuracy/val', acc, epoch)

        writer.add_hparams(params, {'max_accuracy': best_acc})
        writer.flush()


    validate_and_save(model, val_dataloader, writer.log_dir, device, (params['hidden_dim'], 6), args.base_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--base-path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=1024)

    args, _ = parser.parse_known_args()
    main(args)

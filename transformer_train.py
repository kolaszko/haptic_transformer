import argparse
import json
import os
import socket
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import HapticDataset
from models import HAPTR
from utils import save_statistics
from torchsummary import summary


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def main(args):
    params = {
        'num_classes': args.num_classes,
        'projection_dim': args.projection_dim,
        'sequence_length': args.sequence_length,
        'nheads': args.nheads,
        'num_encoder_layers': args.num_encoder_layers,
        'feed_forward': args.feed_forward,
        'dropout': args.dropout,
        'lr': args.lr,
        'gamma': args.gamma,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size
    }

    print(params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_ds = HapticDataset(args.dataset_path, 'train_ds', signal_start=0, signal_length=params['sequence_length'])
    val_ds = HapticDataset(args.dataset_path, 'val_ds', signal_start=0, signal_length=params['sequence_length'])
    test_ds = HapticDataset(args.dataset_path, 'test_ds', signal_start=0, signal_length=params['sequence_length'])

    train_dataloader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=params['batch_size'], shuffle=True)

    results = {}

    torch.manual_seed(42)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(
        'haptr_runs', current_time + '_' + socket.gethostname())

    model = HAPTR(params['num_classes'], params['projection_dim'],
                  params['sequence_length'], params['nheads'],
                  params['num_encoder_layers'], params['feed_forward'],
                  params['dropout'])

    model.to(device)
    summary(model, input_size=(160, 6))

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)

    weight = torch.Tensor(train_ds.weights)
    w = weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    best_acc_val = 0
    best_acc_test = 0

    with SummaryWriter(log_dir=log_dir) as writer:
        print('======== LOG ========')
        print(writer.log_dir)
        print('========    ========')

        with open(os.path.join(log_dir, 'params.json'), 'w') as f:
            f.write(json.dumps(params))

        y_pred_val = []
        y_true_val = []

        y_pred_test = []
        y_true_test = []

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

            y_pred = []
            y_true = []

            model.train(False)
            with torch.no_grad():
                for step, data in enumerate(val_dataloader):
                    s, labels = data[0].to(device), data[1].to(device)

                    out = model(s.unsqueeze(1))
                    loss = criterion(out, labels)

                    _, predicted = torch.max(out.data, 1)

                    y_pred.extend(predicted.data.cpu().numpy())
                    y_true.extend(labels.data.cpu().numpy())

                    correct += (predicted == labels).sum().item()

                    mean_loss += loss.item()

                    print(f'Running loss validation: {loss.item()} in step: {step}')

            acc = (100 * correct / len(val_ds))
            if acc > best_acc_val:
                best_acc_val = acc
                torch.save(model, os.path.join(writer.log_dir, 'val_model'))
                results['val'] = best_acc_val
                y_pred_val = y_pred
                y_true_val = y_true

                print('========== ACC ==========')
                print(best_acc_val)
                print(f'Epoch: {epoch}')
                print('========== === ==========')

            writer.add_scalar('loss/val', mean_loss / len(val_ds), epoch)
            writer.add_scalar('accuracy/val', acc, epoch)

            # Test
            mean_loss = 0.0
            correct = 0

            y_pred = []
            y_true = []

            model.train(False)
            with torch.no_grad():
                for step, data in enumerate(test_dataloader):
                    s, labels = data[0].to(device), data[1].to(device)

                    out = model(s.unsqueeze(1))
                    loss = criterion(out, labels)

                    _, predicted = torch.max(out.data, 1)

                    y_pred.extend(predicted.data.cpu().numpy())
                    y_true.extend(labels.data.cpu().numpy())

                    correct += (predicted == labels).sum().item()

                    mean_loss += loss.item()

                    print(f'Running loss test: {loss.item()} in step: {step}')

            acc = (100 * correct / len(test_ds))
            if acc > best_acc_test:
                best_acc_test = acc
                torch.save(model, os.path.join(writer.log_dir, 'test_model'))
                results['test'] = best_acc_test

                y_pred_test = y_pred
                y_true_test = y_true

                print('========== ACC ==========')
                print(best_acc_test)
                print(f'Epoch: {epoch}')
                print('========== === ==========')

            writer.add_scalar('loss/test', mean_loss / len(test_ds), epoch)
            writer.add_scalar('accuracy/test', acc, epoch)

        save_statistics(y_true_val, y_pred_val, model, os.path.join(log_dir, 'val'), (160, 6))
        save_statistics(y_true_test, y_pred_test, model, os.path.join(log_dir, 'test'), (160, 6))

        writer.flush()

    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        f.write(json.dumps(results))

    results_timer = {}
    dummy_input = torch.randn(1, 160, 6, dtype=torch.float).to(device)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)

    with torch.no_grad():
        for rep in range(repetitions):
            start_time = time.time()
            out = model(dummy_input.unsqueeze(1))
            _, predicted = torch.max(out.data, 1)
            end_time = time.time()
            timings[rep] = (end_time - start_time) * 1000

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)

    results_timer['mean'] = mean_syn
    results_timer['std'] = std_syn
    results_timer['unit'] = 'ms'

    with open(os.path.join(log_dir, 'inference_time.json'), 'w') as f:
        f.write(json.dumps(results_timer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--projection-dim', type=int, default=16)
    parser.add_argument('--sequence-length', type=int, default=160)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--num-encoder-layers', type=int, default=4)
    parser.add_argument('--feed-forward', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=1e-3)

    args, _ = parser.parse_known_args()
    main(args)

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import utils
from models import HAPTR, HAPTR_ModAtt

torch.manual_seed(42)
DEBUG = False


def train(x, y_true, model, criterion, optimizer):
    optimizer.zero_grad()
    y_hat, loss = query(x, y_true, model, criterion)
    loss.backward()
    optimizer.step()
    return y_hat, loss


def query(x, y_true, model, criterion):
    if len(x) == 1:
        x = x[0]

    y_hat, w = model(x)
    loss = criterion(y_hat, y_true)
    return y_hat, loss


def accuracy(hit, num_samples):
    return (100 * hit) / num_samples


def batch_hits(y_hat, y_true):
    _, predicted = torch.max(y_hat.data, 1)
    return (predicted == y_true).sum().item()


def main(args):
    log_dir = utils.log.logdir_name('./', 'haptr_runs')
    utils.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, val_ds, test_ds = utils.dataset.load_dataset(config)
    data_shape = train_ds.signal_length, train_ds.num_modalities, train_ds.mean.shape[-1]
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    results = {}

    # setup a model
    if args.model_type == 'haptr_modatt':
        model = HAPTR_ModAtt(train_ds.num_classes,
                             args.projection_dim, train_ds.signal_length, args.nheads, args.num_encoder_layers,
                             args.feed_forward, args.dropout, train_ds.dim_modalities, train_ds.num_modalities)
    else:
        model = HAPTR(train_ds.num_classes,
                      args.projection_dim, train_ds.signal_length, args.nheads, args.num_encoder_layers,
                      args.feed_forward, args.dropout, train_ds.dim_modalities, train_ds.num_modalities)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=data_shape)

    # setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    w = torch.Tensor(train_ds.weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    # start
    best_epoch_accuracy = 0
    best_acc_test = 0
    y_pred_val, y_true_val = [], []
    y_pred_test, y_true_test = [], []

    with SummaryWriter(log_dir=log_dir) as writer:
        # train/validation
        for epoch in range(args.epochs):
            mean_loss, correct = 0.0, 0
            model.train(True)

            # train loop
            for step, data in enumerate(train_dataloader):
                batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
                out, loss = train(batch_data, batch_labels, model, criterion, optimizer)
                mean_loss += loss.item()
                correct += batch_hits(out, batch_labels)

            # write to the tensorboard
            writer.add_scalar('loss/train', mean_loss / len(train_ds), epoch)
            writer.add_scalar('accuracy/train', accuracy(correct, len(train_ds)), epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            scheduler.step()

            mean_loss, correct = 0.0, 0
            model.train(False)

            # validation loop
            y_pred, y_true = [], []
            with torch.no_grad():
                for step, data in enumerate(val_dataloader):
                    batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
                    out, loss = query(batch_data, batch_labels, model, criterion)
                    mean_loss += loss.item()
                    correct += batch_hits(out, batch_labels)

                    # update statistics
                    _, predicted = torch.max(out.data, 1)
                    y_pred.extend(predicted.data.cpu().numpy())
                    y_true.extend(batch_labels.data.cpu().numpy())

            # calculate epoch accuracy
            epoch_accuracy = accuracy(correct, len(val_ds))
            if epoch_accuracy > best_epoch_accuracy:
                torch.save(model, os.path.join(writer.log_dir, 'val_model'))

                best_epoch_accuracy = epoch_accuracy
                results['val'] = best_epoch_accuracy
                y_pred_val = y_pred
                y_true_val = y_true

            # write to the tensorboard
            writer.add_scalar('loss/val', mean_loss / len(val_ds), epoch)
            writer.add_scalar('accuracy/val', epoch_accuracy, epoch)

            mean_loss, correct = 0.0, 0
            model.train(False)

            # run test loop
            y_pred, y_true = [], []
            with torch.no_grad():
                for step, data in enumerate(test_dataloader):
                    batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
                    out, loss = query(batch_data, batch_labels, model, criterion)
                    mean_loss += loss.item()
                    correct += batch_hits(out, batch_labels)

                    # update statistics
                    _, predicted = torch.max(out.data, 1)
                    y_pred.extend(predicted.data.cpu().numpy())
                    y_true.extend(batch_labels.data.cpu().numpy())

            # calculate epoch accuracy
            epoch_accuracy = accuracy(correct, len(test_ds))
            if epoch_accuracy > best_acc_test:
                torch.save(model, os.path.join(writer.log_dir, 'test_model'))

                best_acc_test = epoch_accuracy
                results['test'] = best_acc_test
                y_pred_test = y_pred
                y_true_test = y_true

                print(f'Epoch {epoch}, test accuracy: {best_acc_test}')

            # update tensorboard
            writer.add_scalar('loss/test', mean_loss / len(test_ds), epoch)
            writer.add_scalar('accuracy/test', epoch_accuracy, epoch)

        utils.log.save_statistics(y_true_val, y_pred_val, model, os.path.join(log_dir, 'val'), data_shape)
        utils.log.save_statistics(y_true_test, y_pred_test, model, os.path.join(log_dir, 'test'), data_shape)
        writer.flush()

    # save all statistics
    utils.log.save_dict(results, os.path.join(log_dir, 'results.txt'))

    # check performance
    timings = utils.torch.measure_interence_time(model, args.repetitions, device)
    results_timer = {
        "mean": np.sum(timings) / args.repetitions,
        "std": np.std(timings),
        "unit": "ms"
    }
    utils.log.save_dict(results_timer, os.path.join(log_dir, 'inference_time.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic_transformer/experiments/config/put_haptr_12.yaml")
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--projection-dim', type=int, default=16)
    parser.add_argument('--sequence-length', type=int, default=160)
    parser.add_argument('--nheads', type=int, default=2)
    parser.add_argument('--num-encoder-layers', type=int, default=3)
    parser.add_argument('--feed-forward', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    parser.add_argument('--repetitions', type=int, default=300)
    parser.add_argument('--model-type', type=str, default='haptr_modatt')

    args, _ = parser.parse_known_args()
    main(args)

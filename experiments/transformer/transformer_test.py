import argparse
import os
import socket
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils

torch.manual_seed(42)

from experiments.transformer.transformer_train import query, accuracy, batch_hits


def main(args):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('haptr_runs_test', current_time + '_' + socket.gethostname())
    os.makedirs(log_dir, exist_ok=True)
    utils.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    split_modalities = True if args.model_type == "haptr_modatt" else False
    _, _, test_ds = utils.dataset.load_dataset(config, split_modalities)
    data_shape = test_ds.signal_length, test_ds.num_modalities, test_ds.mean.shape[-1]
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    results = {}

    # setup a model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path)
    model.eval()
    model.to(device)

    w = torch.Tensor(test_ds.weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    # start
    with SummaryWriter(log_dir=log_dir) as writer:
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
                print(f'Running loss test: {loss.item()} in step: {step}')

        # calculate epoch accuracy
        epoch_accuracy = accuracy(correct, len(test_ds))
        torch.save(model, os.path.join(writer.log_dir, 'test_model'))
        best_acc_test = epoch_accuracy
        results['test'] = best_acc_test
        y_pred_test = y_pred
        y_true_test = y_true

        print('========== ACC ==========')
        print(best_acc_test)
        print('========== === ==========')

        # update tensorboard
        writer.add_scalar('loss/test', mean_loss / len(test_ds), 0)
        writer.add_scalar('accuracy/test', epoch_accuracy, 0)

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
                        default="/home/mbed/Projects/haptic_transformer/experiments/config/config_put.yaml")
    parser.add_argument('--model-path', type=str,
                        default="/home/mbed/Projects/haptic_transformer/haptr_runs/Nov25_17-13-32_mbed/test_model")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--repetitions', type=int, default=300)
    parser.add_argument('--model-type', type=str, default='haptr_modatt')

    args, _ = parser.parse_known_args()
    main(args)

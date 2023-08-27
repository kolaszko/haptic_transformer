import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from experiments.transformer.transformer_train import accuracy, batch_hits

torch.manual_seed(42)


def main(args):
    log_dir = utils.log.logdir_name('./', 'haptr_runs_test')
    utils.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    _, _, test_ds = utils.dataset.load_dataset(config)
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
    # gif = utils.log.GIF(os.path.join(log_dir, 'images'))
    with SummaryWriter(log_dir=log_dir) as writer:
        mean_loss, correct = 0.0, 0
        model.train(False)

        # run test loop
        y_pred, y_true = [], []
        weights = []
        with torch.no_grad():
            for step, data in enumerate(test_dataloader):
                batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)

                out, misc = model(batch_data)
                loss = criterion(out, batch_labels)

                mean_loss += loss.item()
                correct += batch_hits(out, batch_labels)

                # update statistics
                _, predicted = torch.max(out.data, 1)
                y_pred.extend(predicted.data.cpu().numpy())
                y_true.extend(batch_labels.data.cpu().numpy())
                print(f'Running loss test: {loss.item()} in step: {step}')

                # add weights to the gif
                # img = utils.analysis.impose(misc["mod_weights"].cpu().numpy()[0], batch_data.cpu().numpy()[0])
                # gif.add

                weights.append(misc["mod_weights"].cpu().numpy())

        # save resulting gif
        # gif.save()

        # calculate epoch accuracy
        epoch_accuracy = accuracy(correct, len(test_ds))
        torch.save(model, os.path.join(writer.log_dir, 'test_model'))
        best_acc_test = epoch_accuracy
        results['test'] = best_acc_test
        y_pred_test = y_pred
        y_true_test = y_true

        w = weights.mean(0)
        w1, w2 = w[:, 0], w[:, 1]
        wmean1, wmean2 = w1.mean(), w2.mean()
        wmedian1, wmedian2 = np.median(w1), np.median(w2)
        writer.add_scalar('weights/mean/mod_0', wmean1, 0)
        writer.add_scalar('weights/mean/mod_1', wmean2, 0)
        writer.add_scalar('weights/median/mod_0', wmedian1, 0)
        writer.add_scalar('weights/median/mod_1', wmedian2, 0)

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
    parser.add_argument('--dataset-config-file', type=str, default="config/config_put.yaml")
    parser.add_argument('--model-path', type=str, default="test_model")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--repetitions', type=int, default=300)
    parser.add_argument('--model-type', type=str, default='haptr_modatt')

    args, _ = parser.parse_known_args()
    main(args)

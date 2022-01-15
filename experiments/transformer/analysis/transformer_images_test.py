import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import submodules.haptic_transformer.utils as utils

torch.manual_seed(42)


def main(args):
    log_dir = utils.log.logdir_name('./', 'haptr_runs_test_images')
    utils.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    _, _, test_ds = utils.dataset.load_dataset(config)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    # setup a model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path)
    model.eval()
    model.to(device)

    w = torch.Tensor(test_ds.weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    # start
    gif = utils.log.GIF(os.path.join(log_dir, 'images'))
    with SummaryWriter(log_dir=log_dir) as writer:
        mean_loss, correct = 0.0, 0
        model.train(False)

        # run test loop
        with torch.no_grad():
            for step, data in enumerate(test_dataloader):
                batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)

                # add noise to the 0-th modality
                if step == 200:
                    mod = 0
                    mod_size = batch_data[..., mod].size()
                    noise = ((2.0 * torch.rand(size=mod_size) - 1.0)).to(device)
                    num = 100
                    colormaps = [
                        plt.cm.get_cmap('Reds')(np.linspace(0, 1., int(256 / 2))),
                        plt.cm.get_cmap('Blues')(np.linspace(0, 1., int(256 / 2)))
                    ]

                    for i in range(num):
                        x = batch_data.clone()
                        # x[..., mod] += noise * i / num
                        x[..., mod] *= (1.0 - i / num)
                        out, misc = model(x)
                        w = misc["mod_weights"].cpu().numpy()[0]

                        s = x.cpu().numpy()[0]
                        img = utils.analysis.impose(w, s, colormaps)

                        loss = criterion(out, batch_labels)

                        mean_loss += loss.item()
                        gif.add(img)
                        # if batch_hits(out, batch_labels) == 0:
                        #     break

                        w1, w2 = w[:, 0], w[:, 1]
                        wmean1, wmean2 = w1.mean(), w2.mean()
                        wmedian1, wmedian2 = np.median(w1), np.median(w2)
                        writer.add_scalar('weights/mean/mod_0', wmean1, i)
                        writer.add_scalar('weights/mean/mod_1', wmean2, i)
                        writer.add_scalar('weights/median/mod_0', wmedian1, i)
                        writer.add_scalar('weights/median/mod_1', wmedian2, i)
                    gif.save()
                else:
                    continue
                assert 2 == 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic_transformer/experiments/config/config_put.yaml")
    parser.add_argument('--model-path', type=str,
                        default="/home/mbed/Projects/haptic_transformer/experiments/transformer/haptr_runs/Dec02_14-24-18_mbed/test_model")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--repetitions', type=int, default=300)
    parser.add_argument('--model-type', type=str, default='haptr_modatt')

    args, _ = parser.parse_known_args()
    main(args)

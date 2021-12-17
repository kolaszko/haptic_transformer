import argparse

import torch
import yaml
from torch.utils.data import DataLoader

import utils
from experiments.transformer.transformer_train import batch_hits

torch.manual_seed(42)


def main(args):
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
    torch.Tensor(test_ds.weights).to(device)

    # start
    model.train(False)

    # run test loop
    def data_dict(keys):
        return {key: list() for key in keys}

    noise_its = 100
    total_data = dict()
    with torch.no_grad():
        for noised_mod_idx, dim in zip(test_ds.pick_modalities, test_ds.dim_modalities):

            modality_data = data_dict(["weights", "noises", "hits"])
            for step, data in enumerate(test_dataloader):
                batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
                batch_weights, batch_noise, batch_hit = list(), list(), list()

                # start noising one modality
                if type(batch_data) is list:
                    mod_size = batch_data[noised_mod_idx].size()
                else:
                    mod_size = batch_data[..., noised_mod_idx:noised_mod_idx + dim].size()

                bias = 0.0
                noise = ((0.5 * torch.rand(size=mod_size) - 0.5)).to(device)
                for i in range(noise_its):
                    current_bias = bias * i / noise_its
                    current_noise = noise * i / noise_its
                    additive_noise = current_bias + current_noise

                    if type(batch_data) is list:
                        x = [bd.clone() for bd in batch_data]
                        x[noised_mod_idx] += additive_noise
                    else:
                        x = batch_data.clone()
                        x[..., noised_mod_idx:noised_mod_idx + dim] += additive_noise

                    out, misc = model(x)

                    # if classified correctly add to the statistics
                    if "mod_weights" in misc.keys():
                        batch_weights.append(misc["mod_weights"].cpu().numpy())
                    batch_noise.append(current_noise.cpu().numpy())
                    batch_hit.append(batch_hits(out, batch_labels))

                modality_data["weights"].append(batch_weights)
                modality_data["noises"].append(batch_noise)
                modality_data["hits"].append(batch_hit)

            total_data[f"noised_modality_{noised_mod_idx}"] = modality_data
            print(f"Noising modality {noised_mod_idx} finished.")

    utils.log.save_numpy(total_data, f"./{args.suffix}.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-config-file', type=str, default="/home/mbed/Projects/haptic_transformer/experiments/config/put_haptr_12.yaml")
    parser.add_argument('--model-path', type=str, default="/home/mbed/Projects/haptic_transformer/experiments/transformer/HAPTR_vs_HAPTRmodatt/HAPTR/test_model")
    parser.add_argument('--suffix', type=str, default="PUT_HAPTR2")
    # parser.add_argument('--dataset-config-file', type=str, default="/home/mbed/Projects/haptic_transformer/experiments/config/put_split_haptr_12.yaml")
    # parser.add_argument('--model-path', type=str, default="/home/mbed/Projects/haptic_transformer/experiments/transformer/HAPTR_vs_HAPTRmodatt/HAPTR_modatt/test_model")
    # parser.add_argument('--suffix', type=str, default="PUT_HAPTR_modatt2")

    parser.add_argument('--batch-size', type=int, default=32)

    args, _ = parser.parse_known_args()
    main(args)

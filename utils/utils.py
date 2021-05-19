import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score


def summary(model, input_size, batch_size=-1, device="cuda"):
    ''' https://github.com/sksq96/pytorch-summary '''

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    return trainable_params


def save_statistics(y_true, y_pred, model, path, input_size):
    cf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    miou = jaccard_score(y_true, y_pred, average='weighted')

    trainable_params = summary(model, input_size=input_size)

    stat_path = os.path.join(path, 'statistics')

    if not os.path.exists(stat_path):
        os.makedirs(stat_path)

    stats = {
        'acc': accuracy,
        'miou': miou,
        'params': str(trainable_params.cpu().numpy()),
    }

    with open(f'{stat_path}/stats.json', 'w') as f:
        f.write(json.dumps(stats))

    with open(f'{stat_path}/acc.npy', 'wb') as f:
        np.save(f, accuracy)

    with open(f'{stat_path}/cm.npy', 'wb') as f:
        np.save(f, cf_matrix)

    with open(f'{stat_path}/miou.npy', 'wb') as f:
        np.save(f, miou)

    with open(f'{stat_path}/stats.txt', 'w') as f:
        f.write(str(trainable_params.cpu().numpy()))


def validate_and_save(model, ds_loader, name, device, input_size):
    model.train(False)
    y_pred = []
    y_true = []

    with torch.no_grad():
        for step, data in enumerate(ds_loader):
            s, labels = data[0].to(device), data[1].to(device)

            out = model(s.unsqueeze(1))
            _, predicted = torch.max(out.data, 1)
            y_pred.extend(predicted.data.cpu().numpy())

            y_true.extend(labels.data.cpu().numpy())

    cf_matrix = confusion_matrix(y_true, y_pred)

    print(cf_matrix)
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy_score(y_true, y_pred))
    miou = jaccard_score(y_true, y_pred, average='weighted')
    print(jaccard_score(y_true, y_pred, average='weighted'))

    trainable_params = summary(model, input_size=input_size)

    print(name)
    stat_path = os.path.join(name, 'statistics')

    print(stat_path)

    if not os.path.exists(stat_path):
        os.makedirs(stat_path)

    stats = {
        'acc': accuracy,
        'miou': miou,
        'params': str(trainable_params.cpu().numpy()),
    }

    with open(f'{stat_path}/stats.json', 'w') as f:
        f.write(json.dumps(stats))

    with open(f'{stat_path}/acc.npy', 'wb') as f:
        np.save(f, accuracy)

    with open(f'{stat_path}/cm.npy', 'wb') as f:
        np.save(f, cf_matrix)

    with open(f'{stat_path}/miou.npy', 'wb') as f:
        np.save(f, miou)

    with open(f'{stat_path}/stats.txt', 'w') as f:
        f.write(str(trainable_params.cpu().numpy()))

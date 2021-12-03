def determine_dim_size(dim_modalities, pick_modalities):
    if len(pick_modalities) < len(dim_modalities):
        dim_modalities = [dim_modalities[mod_idx] for mod_idx in pick_modalities]
    return dim_modalities

def prepare_batch(multidimensional_ts, split_modalities, pick_modalities, dim_modalities):
    if split_modalities:
        sig = list()
        for mod_idx, mod_dim in zip(pick_modalities, dim_modalities):
            start_idx = mod_idx * mod_dim
            end_idx = mod_idx * mod_dim + mod_dim
            sig.append(multidimensional_ts[..., start_idx:end_idx])

    else:
        sig = [multidimensional_ts]
    return sig

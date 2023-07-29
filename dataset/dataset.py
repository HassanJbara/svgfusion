import torch
import numpy as np
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox
from torch.utils.data import DataLoader
from deepsvg.utils.utils import batchify
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svgtensor_dataset import load_dataset
from torch.utils.data.sampler import SubsetRandomSampler


def dataloader_with_transformed_dataset(vae_model, cfg, batch_n: int, length: int = None):
    dataset = load_dataset(cfg) # the DeepSVG dataset as {'commands': [...], 'args': [...]}
    encoded_dataset_with_labels = []
    data_len = length if length else len(dataset)

    for i in range(data_len):
        xy = dataset.get(i, model_args=['commands', 'args', 'label'])
        label = xy.pop('label')
        encoded_dataset_with_labels.append([encode(xy, vae_model, cfg), label])

    dataset_size = len(encoded_dataset_with_labels)
    batch_size = batch_n
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(encoded_dataset_with_labels, batch_size=batch_size, sampler=train_sampler, drop_last=True,)
    validation_loader = DataLoader(encoded_dataset_with_labels, batch_size=batch_size, sampler=valid_sampler, drop_last=True,)

    return train_loader, validation_loader


def num_classes(dataloader):
    all_classes = set()

    for x, y in dataloader:
          all_classes.update(set(y.numpy()))

    return len(all_classes)


def encode(data, model, cfg):
    device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

    model_args = batchify((data[key] for key in cfg.model_args), device)
    with torch.no_grad():
        z = model(*model_args, encode_mode=True)
        return z.squeeze(dim=0).squeeze(dim=0)

def decode(z, model, do_display=True, return_svg=False, return_png=False):
    commands_y, args_y = model.greedy_sample(z=z)
    tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())
    svg_path_sample = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256), allow_empty=True).normalize().split_paths().set_color("random")

    if return_svg:
        return svg_path_sample

    return svg_path_sample.draw(do_display=do_display, return_png=return_png)
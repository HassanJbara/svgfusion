# svgfusion

<https://github.com/HassanJbara/svgfusion/assets/82707653/746fa72f-892b-4d48-86fd-a3fa672f858a>

This is the code for my bachelor's project, Generating SVGs with Transformer-based Diffusion Models. More details in my blog post.

## Setup

project dependencies:

```bash
pip install -r requirements.txt
```

required linux packages:

```bash
sudo apt-get install libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev
```

download the pretrained model:

```bash
chmod u+x ./pretrained/download.sh
./pretrained/download.sh
```

download the dataset:

```bash
chmod u+x ./dataset/download.sh
./dataset/download.sh
```

## Training

You can train the model with the provided notebooks or use the following command:

```bash
python train.py vae_directory ./pretrained/hierarchical_ordered.pth.tar --batch_size 8 --num_epochs 100 --lr 1e-4 \
        --use_scheduler True --predict_xstart True --depth 28 --n_samples None \
        --wandb True --wandb_key <your_wandb_key> --wandb_project_name svgfusion
```

Those are all the parameters with the default values. None of them is required, except for wandb_key if you want to use wandb, and `n_samples=None` means using all the data in the dataset.

## Notebooks

There are three notebooks in the `notebooks` folder:

- `complete_pipeline.ipynb`: training the model with minimal loggin using the logging function from the `utils` module.
- `complete_pipeline_with_wandb.ipynb`: training the model using wandb for logging and visualization.
- `hyperparameter_sweeping.ipynb`: hyperparameter search using wandb sweeps.

## Sampling

You can sample from the model you saved with the following code:

```python
import torch
from configs.hierarchical_ordered import Config
from utils import load_model, sample_from_diffusion, load_pretrained_autoencoder

cfg = Config()
cfg.data_dir = "../dataset/icons_tensor/" # path to the dataset
cfg.meta_filepath = "../dataset/icons_meta.csv" # path to the meta file
device = "cuda" if torch.cuda.is_available() else "cpu"
deepsvg_path = '../pretrained/hierarchical_ordered.pth.tar' # path to the deepsvg pretrained model
model_path = '../pretrained/my_model.pt' # path to the model you trained

vae_model = load_pretrained_autoencoder(cfg, pretrained_path=deepsvg_path, device=device)
model, diffusion, _ = load_model(model_path, device, for_training=False)

class_ids = [0, 1, 2, 3,] # the class ids of the icons you want to sample from 

samples = sample_from_diffusion(vae_model, diffusion, model, class_ids)
```

The `sample_from_diffusion` method should also draw the SVGs, but in case it doesn't, you can use the following code:

```python
from utils import draw

output_dir = "./samples" # path to the directory where you want to save the samples

for i, sample in enumerate(samples):
    draw(sample, return_png=True, width=1200, height=1200).save(f"{output_dir}/sample_{i}.png")
```

## Credits

The codebase of this project is based mainly on:

- [DeepSVG](https://github.com/alexandre01/deepsvg)
- [Scalable Diffusion Models with Transformers](https://github.com/facebookresearch/DiT)

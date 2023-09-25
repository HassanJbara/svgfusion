# svgfusion

https://github.com/HassanJbara/svgfusion/assets/82707653/519d797d-047c-414d-9db3-e9a37ce19280

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

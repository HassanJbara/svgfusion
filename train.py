import argparse
import json
import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from configs.hierarchical_ordered import Config
from configs.hierarchical_ordered import Config
from utils import load_pretrained_autoencoder, create_model, train, train_with_wandb, save_model
from dataset.dataset import num_classes, dataloader_with_transformed_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vae_directory", type=str, default="./pretrained/hierarchical_ordered.pth.tar", help="path to the pretrained autoencoder")
    
    parser.add_argument("--batch_size", type=int, default=100, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--lr", type=int, default=0.0001, help="learning rate for training")
    parser.add_argument("--use_scheduler", type=bool, default=True, help="use scheduler for training")
    parser.add_argument("--predict_xstart", type=bool, default=True, help="predict xstart for as training target")
    parser.add_argument("--depth", type=int, default=28, help="depth of the created model")
    parser.add_argument("--n_samples", type=int, default=None, help="number of samples to train on. Leave empty for all samples")

    parser.add_argument("--wandb", type=bool, default=True, help="use wandb for logging")
    parser.add_argument("--wandb_key", type=str, default=None, help="wandb api key")
    parser.add_argument("--wandb_project", type=str, default="svgfusion", help="wandb project name")

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args


def main(args):
    if args.n_samples: 
        assert args.n_samples > 0, "n_samples must be greater than 0"
        assert args.n_samples > args.batch_size*2, "n_samples must be at least twice the batch_size"
    
    if args.wandb:
        assert args.wandb_key is not None, "wandb_key must be specified"

    assert args.depth > 0, "depth must be greater than 0"
    assert args.epochs > 0, "epochs must be greater than 0"
    assert args.lr > 0, "learning_rate must be greater than 0"
    assert args.batch_size > 0, "batch_size must be greater than 0"
    assert args.vae_directory is not None, "vae_directory must be specified"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_config = {
        'predict_xstart': args.predict_xstart,
        'learn_sigma': True,
        'use_scheduler': args.use_scheduler,
        'num_heads': 16,
        'depth': args.depth,
        'dropout': 0.1,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'n_samples': args.n_samples, # None = all of the samples
        'magical_number': 0.7128, # mean of std's of latents
    }
    
    # load pretrained autoencoder
    cfg = Config()
    vae_model = load_pretrained_autoencoder(cfg, pretrained_path=args.vae_directory, device=device)

    if args.wandb:
        # tell wandb to get started
        wandb.login(key=args.wandb_key)
        with wandb.init(project=args.wandb_project, config=training_config):
            # access all HPs through wandb.config, so logging matches execution!
            config = wandb.config

            # prepare model, optimizer and dataloaders
            train_dataloader, valid_dataloader = dataloader_with_transformed_dataset(vae_model, cfg, 
                                                                                     batch_n=config.batch_size, 
                                                                                     length=config.n_samples)

            model, diffusion = create_model(dropout=config.dropout, predict_xstart=config.predict_xstart,
                                        n_classes=num_classes(train_dataloader), depth=config.depth, 
                                        learn_sigma=config.learn_sigma, num_heads=config.num_heads)

            optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

            # train the model
            train_with_wandb(model, train_dataloader, diffusion, optimizer, scheduler, config, device=device)

            save_model(model=model, optimizer=optimizer, diffusion=diffusion, scheduler=scheduler, 
                       n_classes=num_classes(train_dataloader), config=config)
    else:
        train_dataloader, valid_dataloader = dataloader_with_transformed_dataset(vae_model, cfg, 
                                                                                 batch_n=config.batch_size, length=config.n_samples)

        model, diffusion = create_model(dropout=config.dropout, predict_xstart=config.predict_xstart,
                                    n_classes=num_classes(train_dataloader), depth=config.depth,
                                    learn_sigma=config.learn_sigma, num_heads=config.num_heads)

        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

        train(model, train_dataloader, diffusion, optimizer, scheduler, config, device=device)

        save_model(model=model, optimizer=optimizer, diffusion=diffusion, scheduler=scheduler, 
                       n_classes=num_classes(train_dataloader), config=config)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
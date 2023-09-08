import IPython.display as ipd
import cairosvg
import io
import os
import wandb
import torch
from torch.optim import AdamW
from datetime import datetime
from pathlib import Path
from deepsvg import utils
from deepsvg.svglib.utils import to_gif
from PIL import Image
from dataset.dataset import decode
from diffusion import create_diffusion
from svgfusion import DiT


def draw(svg_obj, fill=False, file_path=None, do_display=True, return_png=False,
         with_points=False, with_handles=False, with_bboxes=False, with_markers=False, color_firstlast=False,
         with_moves=True, width=600, height=600):
    if file_path is not None:
        _, file_extension = os.path.splitext(file_path)
        if file_extension == ".svg":
            svg_obj.save_svg(file_path)
        elif file_extension == ".png":
            svg_obj.save_png(file_path)
        else:
            raise ValueError(f"Unsupported file_path extension {file_extension}")

    svg_str = svg_obj.to_str(fill=fill, with_points=with_points, with_handles=with_handles, with_bboxes=with_bboxes,
                              with_markers=with_markers, color_firstlast=color_firstlast, with_moves=with_moves)

    if do_display:
        ipd.display(ipd.SVG(svg_str))

    if return_png:
        if file_path is None:
            img_data = cairosvg.svg2png(bytestring=svg_str, output_width=width, output_height=height)
            return Image.open(io.BytesIO(img_data))
        else:
            _, file_extension = os.path.splitext(file_path)

            if file_extension == ".svg":
                img_data = cairosvg.svg2png(url=file_path)
                return Image.open(io.BytesIO(img_data))
            else:
                return Image.open(file_path)



def sample_from_diffusion(vae_model, diffusion, model, class_labels, x_t=None, normalization_factor=0.7, display_gif=False, cfg_scale=4):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_list = []

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 1, 256, device=device) if not x_t else x_t # z = torch.randn(1, 1, 256, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([n] * n, device=device) # [1]
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Sample images:
    if display_gif:
      final_sample = None
      for sample in  diffusion.p_sample_loop_progressive(
          model.forward_with_cfg, z.shape, z, clip_denoised=False,
          model_kwargs=model_kwargs, progress=True, device=device
      ):
        samples, _ = sample["sample"].chunk(2, dim=0)  # Remove null class samples
        # samples = samples * normalization_factor
        sample_svg = decode((samples.unsqueeze(dim=0) / samples.std()) * normalization_factor,
                               vae_model, return_svg=True, do_display=False) #  * normalization_factor
        sample_png = draw(sample_svg, width=1200, height=1200, do_display=False, return_png=True)
        img_list.append(sample_png)
        final_sample = sample

      to_gif(img_list[::2])
      return final_sample

    else:
      samples = diffusion.p_sample_loop(
          model.forward_with_cfg, z.shape, z, clip_denoised=False,
          model_kwargs=model_kwargs, progress=True, device=device
      )

      samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
      # samples = samples * normalization_factor
      decode((samples.unsqueeze(dim=0) / samples.std()) * normalization_factor, vae_model,) # * normalization_factor

      return samples

def log_training(epoch_number: int, loss: float, timestep: int = None):
    Path("./artifacts/").mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    f = open("artifacts/log.txt", "a")

    if timestep: f.write(f"{current_time} Epoch {epoch_number}: {loss} for timestep {timestep} \n\n")
    else: f.write(f"{current_time} Epoch {epoch_number}: {loss} \n\n")

    f.close()

def create_model(predict_xstart=True, dropout=0.1, n_classes=56, depth=28, learn_sigma=True, num_heads=16):

    model = DiT(class_dropout_prob=dropout, num_classes=n_classes, depth=depth, learn_sigma=learn_sigma, num_heads=num_heads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    diffusion = create_diffusion(timestep_respacing="", predict_xstart=predict_xstart,
                                  learn_sigma=learn_sigma)  # default: 1000 steps, linear noise schedule

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    
    return model, diffusion

def save_model(model, optimizer, diffusion, scheduler, n_classes, config):
    export_dir = './models'

    Path(export_dir).mkdir(parents=True, exist_ok=True)

    # will save everything unless this turns out to be heavy on memory
    checkpoint = {
      "model": model.state_dict(),
      "opt": optimizer.state_dict(),
      "diffusion": diffusion,
      "scheduler": scheduler,
      "num_classes": n_classes,
      "config": config,
    }
    exported_model_path = f"{export_dir}/predict_{'x0' if config.predict_xstart else 'noise'}_{config.epochs}.pt"
    torch.save(checkpoint, exported_model_path)


def load_model(model_path, device, for_training=True, return_optimizer=False):
    state = torch.load(model_path, map_location=device)
    
    model = DiT(num_classes=state['num_classes']).to(device)
    model.load_state_dict(state['model'])

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    optimizer.load_state_dict(state['opt'])

    if not for_training:
      model.eval()
      return model, state['diffusion'], state['config']
    else:
      return model, optimizer, state['diffusion'], state['scheduler'], state['config']
    
def load_pretrained_autoencoder(cfg, pretrained_path=None, device=None):
    # Load the autoencoder
    pretrained_path = pretrained_path or "./pretrained/hierarchical_ordered.pth.tar"
    device = device or torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
    
    vae_model = cfg.make_model().to(device)
    utils.load_model(pretrained_path, vae_model)
    vae_model.eval()

    return vae_model

def train_with_wandb(model, train_dataloader, diffusion, optimizer, scheduler, config, device=None):    

    device = device or "cuda" if torch.cuda.is_available() else "cpu"
    steps = 0
    
    for epoch in range(config.epochs):
        avg_loss = 0
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
    
            x = x.squeeze().unsqueeze(dim=1)
            x = x / config.magical_number # mean of std's of latents
    
            model_kwargs = dict(y=y)
    
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            
            wandb.log({"batch_loss": loss.item()}, step=steps)  
            avg_loss += loss.item()
                    
        if config.use_scheduler: scheduler.step(avg_loss / len(train_dataloader))
        wandb.log({"epoch": epoch, "loss": avg_loss / len(train_dataloader), 
                   "learning_rate": optimizer.param_groups[0]['lr']})
        

def train(model, train_dataloader, diffusion, optimizer, scheduler, config, device=None):
    device = device or "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in range(config.epochs):
        avg_loss = 0
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            x = x.squeeze().unsqueeze(dim=1)
            x = x / config.magical_number # mean of std's of latents

            model_kwargs = dict(y=y)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        if config.use_scheduler: scheduler.step(avg_loss / len(train_dataloader))
        print(optimizer.param_groups[0]['lr'])
        log_training(epoch, avg_loss / len(train_dataloader))
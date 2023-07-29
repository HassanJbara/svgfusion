from pathlib import Path
from datetime import datetime
from deepsvg.svglib.utils import to_gif
import IPython.display as ipd
import cairosvg
import io
import os
import torch
from dataset.dataset import decode
from PIL import Image

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

def log_training(epoch_number: int, loss: float, timestep: int = None):
    Path("./artifacts/").mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    f = open("artifacts/log.txt", "a")

    if timestep: f.write(f"{current_time} Epoch {epoch_number}: {loss} for timestep {timestep} \n\n")
    else: f.write(f"{current_time} Epoch {epoch_number}: {loss} \n\n")

    f.close()


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
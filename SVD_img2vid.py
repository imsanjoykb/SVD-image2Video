# Commented out IPython magic to ensure Python compatibility.
#@title <font size="6" color="orange">Stable Video Diffusion img2vid</font>
#@markdown #### [stable-diffusion-art.com](https://stable-diffusion-art.com/stable-video-diffusion-img2vid/)
#@markdown Click the public link `gradio.live` to start
Save_in_Google_Drive = True #@param {type:"boolean"}
Google_Drive_output_path = 'AI_PICS/img2vid' #@param {type:"string"}
Clear_Log = True #@param{type:'boolean'}

# set output path
if Save_in_Google_Drive:
  from google.colab import drive
  drive.mount('/content/drive')
  output_path = '/content/drive/MyDrive/' + Google_Drive_output_path
else:
  output_path = '/content/output'



# %cd /content
import PIL
# Has to use the Image open once for some reason, other jpg upload would fail
!wget https://raw.githubusercontent.com/sagiodev/stable-video-diffusion-img2vid/main/examples/test.jpg
test = PIL.Image.open("/content/test.jpg")
print("Test OK")


!git clone -b dev https://github.com/camenduru/generative-models
!pip install -q -r https://github.com/camenduru/stable-video-diffusion-colab/raw/main/requirements.txt
!pip install -q -e generative-models
!pip install -q -e git+https://github.com/Stability-AI/datapipelines@main#egg=sdata

!apt -y install -qq aria2
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors?download=true -d /content/checkpoints -o svd_xt.safetensors

!mkdir -p /content/scripts/util/detection
!ln -s /content/generative-models/scripts/util/detection/p_head_v1.npz /content/scripts/util/detection/p_head_v1.npz
!ln -s /content/generative-models/scripts/util/detection/w_head_v1.npz /content/scripts/util/detection/w_head_v1.npz

import sys
sys.path.append("generative-models")

import os, math, torch, cv2
from omegaconf import OmegaConf
from glob import glob
from pathlib import Path
from typing import Optional
import numpy as np
from einops import rearrange, repeat

from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as TF
from sgm.util import instantiate_from_config

def load_model(config: str, device: str, num_frames: int, num_steps: int):
    config = OmegaConf.load(config)
    config.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = device
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (num_frames)
    with torch.device(device):
        model = instantiate_from_config(config.model).to(device).eval().requires_grad_(False)
    return model

num_frames = 25
num_steps = 30
model_config = "generative-models/scripts/sampling/configs/svd_xt.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(model_config, device, num_frames, num_steps)
model.conditioner.cpu()
model.first_stage_model.cpu()
model.model.to(dtype=torch.float16)
torch.cuda.empty_cache()
model = model.requires_grad_(False)

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N, T, device, dtype=None):
    batch = {}
    batch_uc = {}
    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device, dtype=dtype)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device, dtype=dtype)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device, dtype=dtype),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]
    if T is not None:
        batch["num_video_frames"] = T
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def sample(
    input_path: str = "/content/test_image.png",
    resize_image: bool = False,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = "/content/outputs",
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    torch.manual_seed(seed)

    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError
    all_out_paths = []
    for input_img_path in all_img_paths:
        with Image.open(input_img_path) as image:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            if resize_image and image.size != (1024, 576):
                print(f"Resizing {image.size} to (1024, 576)")
                image = TF.resize(TF.resize(image, 1024), (576, 1024))
            w, h = image.size
            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                image = image.resize((width, height))
                print(
                    f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )
            image = ToTensor()(image)
            image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )
        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")
        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug
        # low vram mode
        import gc
        gc.collect()
        model.conditioner.cpu()
        model.first_stage_model.cpu()
        torch.cuda.empty_cache()
        model.sampler.verbose = True

        with torch.no_grad():
            with torch.autocast(device):
                model.conditioner.to(device)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )
                model.conditioner.cpu()
                torch.cuda.empty_cache()

                # from here, dtype is fp16
                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)
                for k in uc.keys():
                    uc[k] = uc[k].to(dtype=torch.float16)
                    c[k] = c[k].to(dtype=torch.float16)

                randn = torch.randn(shape, device=device, dtype=torch.float16)
                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(2, num_frames).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                for k in additional_model_inputs:
                    if isinstance(additional_model_inputs[k], torch.Tensor):
                        additional_model_inputs[k] = additional_model_inputs[k].to(dtype=torch.float16)

                def denoiser(input, sigma, c):
                    return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                samples_z.to(dtype=model.first_stage_model.dtype)
                model.en_and_decode_n_samples_a_time = decoding_t
                model.first_stage_model.to(device)
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                model.first_stage_model.cpu()
                torch.cuda.empty_cache()

                os.makedirs(output_folder, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder, "*.mp4")))
                video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    fps_id + 1,
                    (samples.shape[-1], samples.shape[-2]),
                )
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                for frame in vid:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)
                writer.release()
                all_out_paths.append(video_path)
    return all_out_paths

import gradio as gr
import random

def infer(input_path: str, resize_image: bool, n_frames: int, n_steps: int, seed: str, decoding_t: int, motion_bucket_id: int, fps_id: int) -> str:
  if seed == "random":
    seed = random.randint(0, 2**32)
  seed = int(seed)
  print('Seed of this video is: %d'%seed)
  output_paths = sample(
    input_path=input_path,
    resize_image=resize_image,
    num_frames=n_frames,
    num_steps=n_steps,
    fps_id=fps_id,
    motion_bucket_id=127,
    cond_aug=0.02,
    seed=seed,
    decoding_t=decoding_t,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device=device,
    output_folder=output_path
  )
  return output_paths[0]


## GUI
# clear log
if Clear_Log:
    from IPython.display import clear_output
    clear_output()


resize_choices = ["Just resize", "Crop and resize"]
resize_default_index = 1


def clear_image(uploaded_image):
  return None

def resize_image(uploaded_image, resize_option, crop_offset, resized_image_file):
  '''
  Resize the input input based on the resize option.
  Adjust the crop position based on the crop offset
  '''

  if uploaded_image is None:
    return None

  w, h = (1024, 576) # target width and height
  old_h, old_w = uploaded_image.shape[0:2]


  # Crop image
  if resize_option == "Just resize":
    cropped_image = uploaded_image
  elif resize_option == "Crop and resize":
    # crop to same aspect ratio
    if w/h > old_w/old_h:
      target_h = int(old_w * h/w)
      padding1 = int((old_h - target_h)/2)
      padding2 = old_h -padding1
      padding = int(np.interp(crop_offset, [0, 0.5, 1], [0, padding1, padding1 + padding2]))
      cropped_image = uploaded_image[padding:padding+target_h, :, :]
    else:
      target_w = int(old_h * w/h)
      padding1 = int((old_w - target_w)/2)
      padding2 = old_w - padding1
      padding = int(np.interp(crop_offset, [0, 0.5, 1], [0, padding1, padding1 + padding2]))
      cropped_image = uploaded_image[:, padding:padding+target_w, :]
  else:
    ValueError(f"Unknown resize option: {resize_option}")


  # Resize image
  resized_image_file = "/content/resized_image.png"  # cannot be jpg for some reason
  resized_image = cv2.resize(cropped_image, (w, h))
  im = Image.fromarray(resized_image)
  im.save(resized_image_file)
  return resized_image,resized_image_file

with gr.Blocks() as demo:
  resized_image_file = gr.File(visible=False)
  with gr.Column():
    gr.Markdown("# Stable Diffusion Image to Video")

    with gr.Row():
      with gr.Column():
        gr.Markdown("## Upload input image")
        with gr.Row():
          resize_option = gr.Radio(resize_choices, label="Resize mode", value = resize_choices[resize_default_index])
          crop_offset = gr.Slider(0, 1, value=0.5, label="Crop offset")
        input_image = gr.Image(label="Input image")
      with gr.Column():
        gr.Markdown("## Resized image to be used as input")
        resized_image = gr.Image(label="Resized image")

      # trigger on uploading an input image
      input_image.upload(resize_image, inputs =[input_image, resize_option, crop_offset, resized_image_file], outputs = [resized_image, resized_image_file])
      # trigger on clearin the input image
      input_image.clear(clear_image, inputs =[input_image], outputs = resized_image)
      # triggr on changing the resize option
      resize_option.change(resize_image, inputs =[input_image, resize_option, crop_offset, resized_image_file], outputs = [resized_image, resized_image_file])
      # triggr on changing crop offset
      crop_offset.change(resize_image, inputs =[input_image, resize_option, crop_offset, resized_image_file], outputs = [resized_image,resized_image_file])



    btn = gr.Button("Run")
    with gr.Accordion(label="Advanced options", open=False):
      motion_bucket_id = gr.Slider(label="motion bucket id", value=127, minimum = 0, maximum = 255, step = 1, info="Higher for more motion.")
      fps_id = gr.Slider(label="fps id", value=6, minimum = 5, maximum = 30, step = 1, info="frames per second. Increase for faster frame rate.")
      seed = gr.Text(value="random", label="seed (integer or 'random')", info="Fixed integer for generating the same video.")
      n_frames = gr.Number(precision=0, label="number of frames", value=num_frames)
      n_steps = gr.Number(precision=0, label="number of steps", value=num_steps)
      decoding_t = gr.Number(precision=0, label="number of frames decoded at a time", value=2, info="Lower to use less VRAM.")

  with gr.Column():
    video_out = gr.Video(label="generated video")
  examples = [["https://user-images.githubusercontent.com/33302880/284758167-367a25d8-8d7b-42d3-8391-6d82813c7b0f.png"]]
  inputs = [resized_image_file, gr.Checkbox(value=True, visible=False), n_frames, n_steps, seed, decoding_t, motion_bucket_id, fps_id]
  outputs = [video_out]
  btn.click(infer, inputs=inputs, outputs=outputs)
  #gr.Examples(examples=examples, inputs=inputs, outputs=outputs, fn=infer)
  demo.queue().launch(debug=True, share=True, inline=False, show_error=True)


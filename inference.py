import argparse
import copy
import logging
import math
import os
import random
import time
import warnings
from datetime import datetime
from typing import List, Tuple

import diffusers
import mlflow
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torch import nn
from tqdm.auto import tqdm
from icecream import ic

import sys
from pathlib import Path
now_dir = os.path.dirname(__file__)
sys.path.append(now_dir)
from comfy.utils import ProgressBar
from pydub import AudioSegment
from hallo.animate.face_animate import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.datasets.talk_video import TalkingVideoDataset
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.mutual_self_attention import ReferenceAttentionControl
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.utils.util import (compute_snr, delete_additional_ckpt,
                              import_filename, init_output_dir,
                              load_checkpoint, save_checkpoint,
                              seed_everything, tensor_to_video_batch)

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

class Net(nn.Module):
    """
    The Net class defines a neural network model that combines a reference UNet2DConditionModel,
    a denoising UNet3DConditionModel, a face locator, and other components to animate a face in a static image.

    Args:
        reference_unet (UNet2DConditionModel): The reference UNet2DConditionModel used for face animation.
        denoising_unet (UNet3DConditionModel): The denoising UNet3DConditionModel used for face animation.
        face_locator (FaceLocator): The face locator model used for face animation.
        reference_control_writer: The reference control writer component.
        reference_control_reader: The reference control reader component.
        imageproj: The image projection model.
        audioproj: The audio projection model.

    Forward method:
        noisy_latents (torch.Tensor): The noisy latents tensor.
        timesteps (torch.Tensor): The timesteps tensor.
        ref_image_latents (torch.Tensor): The reference image latents tensor.
        face_emb (torch.Tensor): The face embeddings tensor.
        audio_emb (torch.Tensor): The audio embeddings tensor.
        mask (torch.Tensor): Hard face mask for face locator.
        full_mask (torch.Tensor): Pose Mask.
        face_mask (torch.Tensor): Face Mask
        lip_mask (torch.Tensor): Lip Mask
        uncond_img_fwd (bool): A flag indicating whether to perform reference image unconditional forward pass.
        uncond_audio_fwd (bool): A flag indicating whether to perform audio unconditional forward pass.

    Returns:
        torch.Tensor: The output tensor of the neural network model.
    """
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
        reference_control_writer,
        reference_control_reader,
        imageproj,
        audioproj,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.imageproj = imageproj
        self.audioproj = audioproj
    
    def forward(self,):
        """
        empty function to override abstract function of nn Module
        """

    def get_modules(self):
        """
        Simple method to avoid too-few-public-methods pylint error
        """
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
            "face_locator": self.face_locator,
            "imageproj": self.imageproj,
            "audioproj": self.audioproj,
        }
    


def get_attention_mask(mask: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
    """
    Rearrange the mask tensors to the required format.

    Args:
        mask (torch.Tensor): The input mask tensor.
        weight_dtype (torch.dtype): The data type for the mask tensor.

    Returns:
        torch.Tensor: The rearranged mask tensor.
    """
    if isinstance(mask, List):
        _mask = []
        for m in mask:
            _mask.append(
                rearrange(m, "b f 1 h w -> (b f) (h w)").to(weight_dtype))
        return _mask
    mask = rearrange(mask, "b f 1 h w -> (b f) (h w)").to(weight_dtype)
    return mask


def get_noise_scheduler(cfg: argparse.Namespace) -> Tuple[DDIMScheduler, DDIMScheduler]:
    """
    Create noise scheduler for training.

    Args:
        cfg (argparse.Namespace): Configuration object.

    Returns:
        Tuple[DDIMScheduler, DDIMScheduler]: Train noise scheduler and validation noise scheduler.
    """

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    return train_noise_scheduler, val_noise_scheduler


def process_audio_emb(audio_emb: torch.Tensor) -> torch.Tensor:
    """
    Process the audio embedding to concatenate with other tensors.

    Parameters:
        audio_emb (torch.Tensor): The audio embedding tensor to process.

    Returns:
        concatenated_tensors (List[torch.Tensor]): The concatenated tensor list.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0] - 1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb

def cut_audio(audio_path, save_dir, length=60):
    audio = AudioSegment.from_wav(audio_path)

    segment_length = length * 1000 # pydub使用毫秒

    num_segments = len(audio) // segment_length + (1 if len(audio) % segment_length != 0 else 0)

    os.makedirs(save_dir, exist_ok=True)

    audio_list = [] 

    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, len(audio))
        segment = audio[start_time:end_time]
        
        path = f"{save_dir}/segment_{i+1}.wav"
        audio_list.append(path)
        segment.export(path, format="wav")

    return audio_list

def log_validation(
    accelerator: Accelerator,
    vae: AutoencoderKL,
    net: Net,
    scheduler: DDIMScheduler,
    generator: torch.Generator = None,
    cfg: dict = None,
    times: int = None,
    face_analysis_model_path: str = "",
) -> None:
    """
    Log validation video during the training process.

    Args:
        accelerator (Accelerator): The accelerator for distributed training.
        vae (AutoencoderKL): The autoencoder model.
        net (Net): The main neural network model.
        scheduler (DDIMScheduler): The scheduler for noise.
        width (int): The width of the input images.
        height (int): The height of the input images.
        clip_length (int): The length of the video clips. Defaults to 24.
        generator (torch.Generator): The random number generator. Defaults to None.
        cfg (dict): The configuration dictionary. Defaults to None.
        save_dir (str): The directory to save validation results. Defaults to None.
        global_step (int): The current global step in training. Defaults to 0.
        times (int): The number of inference times. Defaults to None.
        face_analysis_model_path (str): The path to the face analysis model. Defaults to "".

    Returns:
        torch.Tensor: The tensor result of the validation.
    """
    source_image_path = cfg.source_image
    driving_audio_path = cfg.driving_audio

    save_path = os.path.join(cfg.save_path, Path(source_image_path).stem)
    save_seg_path = os.path.join(save_path, "seg_video")
    print("save path: ", save_path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_seg_path):
        os.makedirs(save_seg_path)
    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    face_locator = ori_net.face_locator
    imageproj = ori_net.imageproj
    audioproj = ori_net.audioproj

    generator = torch.manual_seed(42)
    tmp_denoising_unet = copy.deepcopy(denoising_unet)

    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=tmp_denoising_unet,
        face_locator=face_locator,
        image_proj=imageproj,
        scheduler=scheduler,
    )
    pipeline = pipeline.to("cuda")

    img_size = (cfg.data.source_image.width,
                cfg.data.source_image.height)
    clip_length = cfg.data.n_sample_frames
    with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
        source_image_pixels, \
        source_image_face_region, \
        source_image_face_emb, \
        source_image_full_mask, \
        source_image_face_mask, \
        source_image_lip_mask = image_processor.preprocess(
            source_image_path, save_path, cfg.face_expand_ratio)

    
    # 3.2 prepare audio embeddings
    sample_rate = cfg.data.driving_audio.sample_rate
    assert sample_rate == 16000, "audio sample rate must be 16000"
    fps = cfg.data.export_video.fps
    wav2vec_model_path = cfg.wav2vec.model_path
    wav2vec_only_last_features = cfg.wav2vec.features == "last"
    audio_separator_model_file = cfg.audio_separator.model_path

    
    if cfg.use_cut:
        audio_list = cut_audio(driving_audio_path, os.path.join(
            save_path, f"seg-long-{Path(driving_audio_path).stem}"))

        audio_emb_list = []
        l = 0

        audio_processor = AudioProcessor(
                sample_rate,
                fps,
                wav2vec_model_path,
                wav2vec_only_last_features,
                os.path.dirname(audio_separator_model_file),
                os.path.basename(audio_separator_model_file),
                os.path.join(save_path, "audio_preprocess")
            )
        
        for idx, audio_path in enumerate(audio_list):
            padding = (idx+1) == len(audio_list)
            emb, length = audio_processor.preprocess(audio_path, clip_length, 
                                                     padding=padding, processed_length=l)
            audio_emb_list.append(emb)
            l += length
        
        audio_emb = torch.cat(audio_emb_list)
        audio_length = l
    
    else:
        with AudioProcessor(
                sample_rate,
                fps,
                wav2vec_model_path,
                wav2vec_only_last_features,
                os.path.dirname(audio_separator_model_file),
                os.path.basename(audio_separator_model_file),
                os.path.join(save_path, "audio_preprocess")
            ) as audio_processor:
                audio_emb, audio_length = audio_processor.preprocess(driving_audio_path, clip_length)

    audio_emb = process_audio_emb(audio_emb)

    source_image_pixels = source_image_pixels.unsqueeze(0)
    source_image_face_region = source_image_face_region.unsqueeze(0)
    source_image_face_emb = source_image_face_emb.reshape(1, -1)
    source_image_face_emb = torch.tensor(source_image_face_emb)

    source_image_full_mask = [
        (mask.repeat(clip_length, 1))
        for mask in source_image_full_mask
    ]
    source_image_face_mask = [
        (mask.repeat(clip_length, 1))
        for mask in source_image_face_mask
    ]
    source_image_lip_mask = [
        (mask.repeat(clip_length, 1))
        for mask in source_image_lip_mask
    ]


    times = audio_emb.shape[0] // clip_length

    tensor_result = []

    generator = torch.manual_seed(42)

    ic(audio_emb.shape)
    ic(audio_length)    
    batch_size = 60
    start = 0
    comfy_par = ProgressBar(times)
    for t in range(times):
        print(f"[{t+1}/{times}]")

        if len(tensor_result) == 0:
            # The first iteration
            motion_zeros = source_image_pixels.repeat(
                cfg.data.n_motion_frames, 1, 1, 1)
            motion_zeros = motion_zeros.to(
                dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat(
                [source_image_pixels, motion_zeros], dim=0)  # concat the ref image and the first motion frames
        else:
            motion_frames = tensor_result[-1][0]
            motion_frames = motion_frames.permute(1, 0, 2, 3)
            motion_frames = motion_frames[0-cfg.data.n_motion_frames:]
            motion_frames = motion_frames * 2.0 - 1.0
            motion_frames = motion_frames.to(
                dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat(
                [source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames
        
        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

        pixel_motion_values = pixel_values_ref_img[:, 1:]

        if cfg.use_mask:
            b, f, c, h, w = pixel_motion_values.shape
            rand_mask = torch.rand(h, w)
            mask = rand_mask > cfg.mask_rate
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
            mask = mask.expand(b, f, c, h, w)  

            face_mask = source_image_face_region.repeat(f, 1, 1, 1).unsqueeze(0)
            assert face_mask.shape == mask.shape
            mask = mask | face_mask.bool()

            pixel_motion_values = pixel_motion_values * mask
            pixel_values_ref_img[:, 1:] = pixel_motion_values

        
        assert pixel_motion_values.shape[0] == 1

        audio_tensor = audio_emb[
            t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])
        ]
        audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(
            device=net.audioproj.device, dtype=net.audioproj.dtype)
        audio_tensor = net.audioproj(audio_tensor)
        motion_scale = [cfg.pose_weight, cfg.face_weight, cfg.lip_weight]
        pipeline_output = pipeline(
            ref_image=pixel_values_ref_img,
            audio_tensor=audio_tensor,
            face_emb=source_image_face_emb,
            face_mask=source_image_face_region,
            pixel_values_full_mask=source_image_full_mask,
            pixel_values_face_mask=source_image_face_mask,
            pixel_values_lip_mask=source_image_lip_mask,
            width=img_size[0],
            height=img_size[1],
            video_length=clip_length,
            num_inference_steps=cfg.inference_steps,
            guidance_scale=cfg.cfg_scale,
            generator=generator,
            motion_scale=motion_scale,
        )

        ic(pipeline_output.videos.shape)
        tensor_result.append(pipeline_output.videos)

        if (t+1) % batch_size == 0 or (t+1)==times:
            last_motion_frame = [tensor_result[-1]]
            ic(len(tensor_result))

            if start!=0:
                tensor_result = torch.cat(tensor_result[1:], dim=2)
            else:
                tensor_result = torch.cat(tensor_result, dim=2)
            
            tensor_result = tensor_result.squeeze(0)
            f = tensor_result.shape[1]
            length = min(f, audio_length)
            tensor_result = tensor_result[:, :length]

            ic(tensor_result.shape)
            ic(start)
            ic(audio_length)

            name = Path(save_path).name
            output_file = os.path.join(save_seg_path, f"{name}-{t+1:06}.mp4")

            tensor_to_video_batch(tensor_result, output_file, start, driving_audio_path)
            del tensor_result

            tensor_result = last_motion_frame
            audio_length -= length
            start += length
        comfy_par.update(1)
    
    # clean up
    del tmp_denoising_unet
    del pipeline
    del image_processor
    del audio_processor
    torch.cuda.empty_cache()
    return save_seg_path


def inference_process(cfg) -> None:
    """
    Trains the model using the given configuration (cfg).

    Args:
        cfg (dict): The configuration dictionary containing the parameters for training.

    Notes:
        - This function trains the model using the given configuration.
        - It initializes the necessary components for training, such as the pipeline, optimizer, and scheduler.
        - The training progress is logged and tracked using the accelerator.
        - The trained model is saved after the training is completed.
    """
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # create output dir for training
    exp_name = "test"
    save_dir = cfg.save_path
    validation_dir = save_dir
    if accelerator.is_main_process:
        init_output_dir([save_dir])

    accelerator.wait_for_everyone()

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    # Create Models
    vae = AutoencoderKL.from_pretrained(cfg.vae.model_path).to(
        "cuda", dtype=weight_dtype
    )
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            cfg.unet_additional_kwargs),
        use_landmark=False
    ).to(device="cuda", dtype=weight_dtype)
    imageproj = ImageProjModel(
        cross_attention_dim=denoising_unet.config.cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    ).to(device="cuda", dtype=weight_dtype)
    face_locator = FaceLocator(
        conditioning_embedding_channels=320,
    ).to(device="cuda", dtype=weight_dtype)
    audioproj = AudioProjModel(
        seq_len=5,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ).to(device="cuda", dtype=weight_dtype)

    # Freeze
    vae.requires_grad_(False)
    imageproj.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    face_locator.requires_grad_(False)
    audioproj.requires_grad_(False)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        face_locator,
        reference_control_writer,
        reference_control_reader,
        imageproj,
        audioproj,
    ).to(dtype=weight_dtype)

    m,u = net.load_state_dict(
        torch.load(
            os.path.join(cfg.audio_ckpt_dir, f"net.pth"),
            map_location="cpu",
        ),
    )
    assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
    print("loaded weight from ", os.path.join(cfg.audio_ckpt_dir))
    '''
    if cfg.if_fp8:
        from .fp8_optimization import convert_fp8_linear
        convert_fp8_linear(net,original_dtype=torch.bfloat16)
    '''
    
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()

        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
    
    net = accelerator.prepare(net)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )

    logger.info("***** Running Inferencing *****")

    # Inference
    save_seg_path=log_validation(
        accelerator=accelerator,
        vae=vae,
        net=net,
        scheduler=val_noise_scheduler,
        cfg=cfg,
        face_analysis_model_path=cfg.face_analysis.model_path
    )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return save_seg_path


def load_config(config_path: str) -> dict:
    """
    Loads the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """

    if config_path.endswith(".yaml"):
        return OmegaConf.load(config_path)
    if config_path.endswith(".py"):
        return import_filename(config_path).cfg
    raise ValueError("Unsupported format for config file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/inference/inference.yaml"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        inference_process(config)
    except Exception as e:
        logging.error("Failed to execute the training process: %s", e)

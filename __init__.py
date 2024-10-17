import os,sys
import folder_paths
import os.path as osp
now_dir = osp.dirname(osp.abspath(__file__))
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")
hallo_dir = osp.join(aifsh_dir,"HALLO")
sys.path.append(now_dir)
from huggingface_hub import snapshot_download
output_dir = folder_paths.get_output_directory()
save_dir = osp.join(output_dir,"hallo2")
cache_dir = osp.join(save_dir,"cache")
config_file = osp.join(now_dir,"long.yaml")

import time
import shutil
import torchaudio
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from .inference_long import inference_process
from hallo.utils.util import merge_videos

class Hallo2Node:
    def __init__(self):
        if not osp.exists(osp.join(hallo_dir,"hallo2","net.pth")):
            snapshot_download(repo_id="fudan-generative-ai/hallo2",local_dir=hallo_dir)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "source_image":("IMAGE",),
                "driving_audio":("AUDIO",),
                "inference_steps":("INT",{
                    "default": 40,
                    "step":1,
                    "min":5,
                    "max":100,
                    "display":"slider"
                }),
                "use_cut":("BOOLEAN",{
                    "default": False,
                    "tooltip":"cut audio into 1 mintues piece"
                }),
                "pose_weight":("FLOAT",{
                    "default":1.0
                }),
                "face_weight":("FLOAT",{
                    "default":1.0
                }),
                "lip_weight":("FLOAT",{
                    "default":1.0
                }),
                "face_expand_ratio":("FLOAT",{
                    "default":1.2
                }),
                "seed":("INT",{
                    "default":42
                })
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_hallo2"

    def gen_video(self,source_image,driving_audio,inference_steps,
                  use_cut,pose_weight,face_weight,lip_weight,
                  face_expand_ratio,seed):
        shutil.rmtree(save_dir,ignore_errors=True)
        os.makedirs(save_dir,exist_ok=True)
        config = OmegaConf.load(config_file)
        source_image_path = osp.join(save_dir,"source_image.jpg")
        img_np = source_image.numpy()[0] * 255
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        img_pil.save(source_image_path)
        config.source_image = source_image_path
        driving_audio_path = osp.join(save_dir,"audio.wav")
        waveform = driving_audio['waveform'].squeeze(0)
        org_sr = driving_audio['sample_rate']
        target_sr = 16000
        if org_sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=org_sr,new_freq=target_sr)(waveform)
        torchaudio.save(driving_audio_path,waveform,target_sr)
        config.driving_audio = driving_audio_path

        config.audio_ckpt_dir = osp.join(hallo_dir,"hallo2")

        config.save_path = save_dir
        config.cache_path = cache_dir
        config.base_model_path = osp.join(hallo_dir,"stable-diffusion-v1-5")
        config.motion_module_path = osp.join(hallo_dir,"motion_module/mm_sd_v15_v2.ckpt")
        config.face_analysis.model_path = osp.join(hallo_dir,"face_analysis")
        config.wav2vec.model_path = osp.join(hallo_dir,"wav2vec/wav2vec2-base-960h")
        config.audio_separator.model_path = osp.join(hallo_dir,"audio_separator/Kim_Vocal_2.onnx")
        config.vae.model_path = osp.join(hallo_dir,"sd-vae-ft-mse")

        config.seed = seed
        config.inference_steps = inference_steps
        config.use_cut = use_cut
        config.pose_weight = pose_weight
        config.face_weight = face_weight
        config.lip_weight = lip_weight
        config.face_expand_ratio = face_expand_ratio
        save_seg_path = inference_process(config)
        out_video = osp.join(Path(save_seg_path).parent, "merge_video.mp4")
        merge_videos(save_seg_path,out_video)
        time_name = time.time_ns()
        video_name = f"hallo2_{time_name}"
        res_video = osp.join(output_dir,f"{video_name}.mp4")
        shutil.copy(out_video,res_video)
        # shutil.rmtree(save_dir,ignore_errors=True)
        return (res_video,)


class Hallo2UpscaleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "input_path":("VIDEO",),
                "fidelity_weight":("FLOAT",{
                    "default":0.5,
                    "tooltip":"Balance the quality and fidelity. Default: 0.5"
                }),
                "upscale":("INT",{
                    "default":2,
                    "tooltip":"The final upsampling scale of the image. Default: 2"
                }),
                "detection_model":(["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
            }
        }
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_hallo2"

    def gen_video(self,input_path,fidelity_weight,upscale,detection_model):
        video_name = video_name = os.path.basename(input_path)[:-4]
        result_root = osp.join(save_dir,f'hq_results/{video_name}_{fidelity_weight}_{upscale}')
        os.makedirs(result_root,exist_ok=True)
        py =  sys.executable or "python"
        cmd = f"""{py} {osp.join(now_dir,"video_sr.py")} -i {input_path} -o {result_root} \
        -w {fidelity_weight} -s {upscale} --detection_model {detection_model} \
              --bg_upsampler "realesrgan" --face_upsample """
        print(cmd)
        os.system(cmd)
        res_video = osp.join(output_dir,f"{video_name}.mp4")
        shutil.copy(osp.join(result_root,f"{video_name}.mp4"), res_video)
        return (res_video,)



NODE_CLASS_MAPPINGS = {
    "Hallo2Node": Hallo2Node,
    "Hallo2UpscaleNode":Hallo2UpscaleNode,
}
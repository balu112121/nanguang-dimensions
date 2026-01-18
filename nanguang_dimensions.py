import torch
import numpy as np
from PIL import Image, ImageOps
import json
import os

# ComfyUI 导入
import folder_paths
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.model_management
from comfy.cli_args import args
from nodes import common_ksampler

# 预设尺寸配置
PRESET_SIZES = {
    "自定义尺寸": {"width": 512, "height": 512, "ratio": "1:1"},
    "512x512px      比例1:1(SD1.5-最小)": {"width": 512, "height": 512, "ratio": "1:1"},
    "768x768px      比例1:1(SD1.5-最大)": {"width": 768, "height": 768, "ratio": "1:1"},
    "512x768px      比例2:3(SDXL-最小)": {"width": 512, "height": 768, "ratio": "2:3"},
    "1024x1024px  比例1:1(SDXL-最大)": {"width": 1024, "height": 1024, "ratio": "1:1"},
    "256x256px      比例1:1(FLUX.1-最小)": {"width": 256, "height": 256, "ratio": "1:1"},
    "1440x1440px  比例1:1(FLUX.1-最大)": {"width": 1440, "height": 1440, "ratio": "1:1"},
    "800x800px      比例1:1(电商主图)": {"width": 800, "height": 800, "ratio": "1:1"},
    "750x1000px    比例3:4(标准)": {"width": 750, "height": 1000, "ratio": "3:4"},
    "800x1200px    比例2:3(照片)": {"width": 800, "height": 1200, "ratio": "2:3"},
    "295x413px      比例2:3(一寸照片)": {"width": 295, "height": 413, "ratio": "2:3"},
    "413x579px      比例2:3(二寸照片)": {"width": 413, "height": 579, "ratio": "2:3"},
    "1795x1205px  比例6:4(六寸寸照片)": {"width": 1795, "height": 1205, "ratio": "6:4"},
    "1440x1440px  比例1:1(正方形)": {"width": 1440, "height": 1440, "ratio": "1:1"},
    "1080x1920px  比例9:16(移动设备)": {"width": 1080, "height": 1920, "ratio": "9:16"},
    "1920x1080px  比例16:9(高清视频)": {"width": 1920, "height": 1080, "ratio": "16:9"},
    "1920x1440px  比例4:3(PPT)": {"width": 1920, "height": 1440, "ratio": "4:3"},
    "1920x823px    比例21:9(超宽)": {"width": 1920, "height": 823, "ratio": "21:9"},
    "1920x960px    比例2:1(全景)": {"width": 1920, "height": 960, "ratio": "2:1"},
    "1920x1536px  比例5:4(经典)": {"width": 1920, "height": 1536, "ratio": "5:4"},
}

# 缩放算法映射
SCALING_METHODS = {
    "最近邻": Image.NEAREST,
    "双线性": Image.BILINEAR,
    "双三次": Image.BICUBIC,
    "Lanczos": Image.LANCZOS,
    "盒状": Image.BOX,
    "汉明": Image.HAMMING,
}

# 裁剪方式
CROP_METHODS = ["居中裁剪", "边缘裁剪", "填充", "拉伸"]

class NanguangImageDimensions:
    """南光图像尺寸节点 - 处理图像尺寸调整"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset_size": (list(PRESET_SIZES.keys()), {"default": "512x512px      比例1:1(SD1.5-最小)"}),
                "enable_custom": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "crop_method": (CROP_METHODS, {"default": "居中裁剪"}),
                "scaling_method": (list(SCALING_METHODS.keys()), {"default": "Lanczos"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "遮罩")
    FUNCTION = "process"
    CATEGORY = "南光尺寸"
    
    def process(self, image, preset_size, enable_custom, custom_width, custom_height, 
                batch_size, crop_method, scaling_method, mask=None):
        
        # 获取目标尺寸
        if enable_custom:
            width = custom_width
            height = custom_height
        else:
            preset = PRESET_SIZES[preset_size]
            width = preset["width"]
            height = preset["height"]
        
        # 确保尺寸是8的倍数（兼容大多数模型）
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        # 调整图像尺寸
        result_images = []
        result_masks = []
        
        for i in range(min(batch_size, len(image))):
            img = image[i]
            
            # 转换为PIL图像
            img_np = 255. * img.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            
            # 根据裁剪方式调整尺寸
            if crop_method == "居中裁剪":
                # 计算缩放比例，使图像至少覆盖目标尺寸
                scale = max(width / img_pil.width, height / img_pil.height)
                new_width = int(img_pil.width * scale)
                new_height = int(img_pil.height * scale)
                img_pil = img_pil.resize((new_width, new_height), SCALING_METHODS[scaling_method])
                
                # 居中裁剪
                left = (new_width - width) // 2
                top = (new_height - height) // 2
                img_pil = img_pil.crop((left, top, left + width, top + height))
                
            elif crop_method == "边缘裁剪":
                # 计算缩放比例，使图像适应目标尺寸
                scale = min(width / img_pil.width, height / img_pil.height)
                new_width = int(img_pil.width * scale)
                new_height = int(img_pil.height * scale)
                img_pil = img_pil.resize((new_width, new_height), SCALING_METHODS[scaling_method])
                
                # 边缘裁剪（从左上角开始）
                img_pil = img_pil.crop((0, 0, min(width, new_width), min(height, new_height)))
                
            elif crop_method == "填充":
                # 调整到目标尺寸
                img_pil = img_pil.resize((width, height), SCALING_METHODS[scaling_method])
                
            else:  # 拉伸
                # 拉伸到目标尺寸
                img_pil = img_pil.resize((width, height), SCALING_METHODS[scaling_method])
            
            # 转换回tensor
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,]
            result_images.append(img_tensor)
            
            # 处理遮罩（如果有）
            if mask is not None and i < len(mask):
                mask_img = mask[i].unsqueeze(0)
                mask_np = mask_img.cpu().numpy() * 255
                mask_pil = Image.fromarray(np.clip(mask_np[0], 0, 255).astype(np.uint8))
                
                # 使用相同的尺寸调整方法处理遮罩
                if crop_method == "居中裁剪":
                    scale = max(width / mask_pil.width, height / mask_pil.height)
                    new_width = int(mask_pil.width * scale)
                    new_height = int(mask_pil.height * scale)
                    mask_pil = mask_pil.resize((new_width, new_height), SCALING_METHODS[scaling_method])
                    left = (new_width - width) // 2
                    top = (new_height - height) // 2
                    mask_pil = mask_pil.crop((left, top, left + width, top + height))
                elif crop_method == "边缘裁剪":
                    scale = min(width / mask_pil.width, height / mask_pil.height)
                    new_width = int(mask_pil.width * scale)
                    new_height = int(mask_pil.height * scale)
                    mask_pil = mask_pil.resize((new_width, new_height), SCALING_METHODS[scaling_method])
                    mask_pil = mask_pil.crop((0, 0, min(width, new_width), min(height, new_height)))
                elif crop_method == "填充":
                    mask_pil = mask_pil.resize((width, height), SCALING_METHODS[scaling_method])
                else:  # 拉伸
                    mask_pil = mask_pil.resize((width, height), SCALING_METHODS[scaling_method])
                
                mask_np = np.array(mask_pil).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_np)[None,]
                result_masks.append(mask_tensor)
        
        # 合并结果
        if len(result_images) > 0:
            result_image = torch.cat(result_images, dim=0)
        else:
            result_image = image
            
        if len(result_masks) > 0:
            result_mask = torch.cat(result_masks, dim=0)
        else:
            result_mask = mask if mask is not None else torch.zeros((1, height, width))
        
        return (result_image, result_mask)

class NanguangEmptyLatentDimensions:
    """南光空Latent图像尺寸节点 - 生成空Latent"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_size": (list(PRESET_SIZES.keys()), {"default": "512x512px      比例1:1(SD1.5-最小)"}),
                "enable_custom": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜在空间",)
    FUNCTION = "generate"
    CATEGORY = "南光尺寸"
    
    def generate(self, preset_size, enable_custom, custom_width, custom_height, batch_size):
        
        # 获取目标尺寸
        if enable_custom:
            width = custom_width
            height = custom_height
        else:
            preset = PRESET_SIZES[preset_size]
            width = preset["width"]
            height = preset["height"]
        
        # 确保尺寸是8的倍数
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        # 生成空latent
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        
        return ({"samples": latent},)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "NanguangImageDimensions": NanguangImageDimensions,
    "NanguangEmptyLatentDimensions": NanguangEmptyLatentDimensions,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "NanguangImageDimensions": "南光图像尺寸",
    "NanguangEmptyLatentDimensions": "南光空Latent图像尺寸",
}
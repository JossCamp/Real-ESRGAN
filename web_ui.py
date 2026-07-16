"""
Web UI for Real-ESRGAN Training and Inference
A Gradio-based interface for comfortable training and inference in the browser
"""

import argparse
import os
import sys
import tempfile
import threading
from pathlib import Path

import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2

# Import Real-ESRGAN components
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class TrainingProgress:
    """Track training progress"""
    def __init__(self):
        self.iteration = 0
        self.total_iterations = 0
        self.loss_g = 0
        self.loss_d = 0
        self.status = "idle"
        self.should_stop = False
        
    def stop(self):
        self.should_stop = True


training_progress = TrainingProgress()
current_model = None
current_upsampler = None


def get_device_info():
    """Get information about available hardware acceleration"""
    info = []
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        info.append(f"✓ CUDA Available: {gpu_count} GPU(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            info.append(f"  - GPU {i}: {gpu_name}")
            # Check for Tensor Core support
            capability = torch.cuda.get_device_capability(i)
            if capability[0] >= 7:
                info.append(f"    ✓ Tensor Cores supported (Compute Capability {capability[0]}.{capability[1]})")
            else:
                info.append(f"    ℹ Compute Capability {capability[0]}.{capability[1]}")
    else:
        info.append("✗ CUDA not available - using CPU")
    return "\n".join(info)


def load_model(model_name, denoise_strength=0.5, use_fp16=True, gpu_id=None):
    """Load a Real-ESRGAN model"""
    global current_model, current_upsampler
    
    models_config = {
        'RealESRGAN_x4plus': {
            'model': lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'scale': 4,
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        },
        'RealESRNet_x4plus': {
            'model': lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'scale': 4,
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'
        },
        'RealESRGAN_x4plus_anime_6B': {
            'model': lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
            'scale': 4,
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
        },
        'RealESRGAN_x2plus': {
            'model': lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
            'scale': 2,
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        },
        'realesr-animevideov3': {
            'model': lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'),
            'scale': 4,
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'
        },
        'realesr-general-x4v3': {
            'model': lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
            'scale': 4,
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        }
    }
    
    if model_name not in models_config:
        return None, f"Model {model_name} not found"
    
    config = models_config[model_name]
    model = config['model']()
    netscale = config['scale']
    
    # Load model path
    model_path = os.path.join('weights', f'{model_name}.pth')
    if not os.path.isfile(model_path):
        try:
            model_path = load_file_from_url(
                url=config['url'], 
                model_dir=os.path.join(ROOT_DIR, 'weights'), 
                progress=True, 
                file_name=None
            )
        except Exception as e:
            return None, f"Error downloading model: {str(e)}"
    
    # Handle DNI for realesr-general-x4v3
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        if os.path.exists(wdn_model_path):
            model_path = [model_path, wdn_model_path]
            dni_weight = [denoise_strength, 1 - denoise_strength]
    
    try:
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=256,  # Default tile size for memory efficiency
            tile_pad=10,
            pre_pad=0,
            half=use_fp16 and torch.cuda.is_available(),  # Use FP16 when available
            gpu_id=gpu_id
        )
        current_model = model_name
        current_upsampler = upsampler
        return upsampler, f"✓ Model loaded successfully: {model_name}"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def enhance_image(image, model_name, denoise_strength, use_fp16, tile_size, outscale):
    """Enhance a single image"""
    if image is None:
        return None, "No image provided"
    
    # Convert PIL to numpy
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Load or get current upsampler
    global current_upsampler
    if current_upsampler is None or current_model != model_name:
        upsampler, msg = load_model(model_name, denoise_strength, use_fp16)
        if upsampler is None:
            return None, msg
    else:
        upsampler = current_upsampler
    
    try:
        # Update tile size if needed
        upsampler.tile_size = tile_size
        
        output, _ = upsampler.enhance(img_np, outscale=outscale)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output), f"✓ Enhancement complete! Output size: {output.shape[1]}x{output.shape[0]}"
    except Exception as e:
        return None, f"Error during enhancement: {str(e)}"


def enhance_batch(input_folder, output_folder, model_name, denoise_strength, use_fp16, tile_size, outscale):
    """Enhance multiple images from a folder"""
    if not os.path.exists(input_folder):
        return "Input folder does not exist"
    
    # Load or get current upsampler
    global current_upsampler
    if current_upsampler is None or current_model != model_name:
        upsampler, msg = load_model(model_name, denoise_strength, use_fp16)
        if upsampler is None:
            return msg
    else:
        upsampler = current_upsampler
    
    upsampler.tile_size = tile_size
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']
    image_files = [f for f in os.listdir(input_folder) 
                   if os.path.isfile(os.path.join(input_folder, f)) and 
                   os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        return "No images found in input folder"
    
    processed = 0
    errors = 0
    
    for idx, filename in enumerate(image_files):
        try:
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                print(f"Failed to read: {filename}")
                errors += 1
                continue
            
            output, _ = upsampler.enhance(img, outscale=outscale)
            
            # Save with same name
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, output)
            processed += 1
            print(f"Processed {idx+1}/{len(image_files)}: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            errors += 1
    
    return f"Batch complete: {processed} images processed, {errors} errors"


def start_training(config_file, resume_state, use_tensor_cores, num_gpu):
    """Start training process"""
    global training_progress
    
    if not os.path.exists(config_file):
        return "Configuration file not found"
    
    training_progress.should_stop = False
    training_progress.status = "running"
    
    # Prepare command
    cmd = f"python realesrgan/train.py -opt {config_file}"
    
    if resume_state:
        cmd += f" --resume {resume_state}"
    
    # Set environment variables for Tensor Core optimization
    env = os.environ.copy()
    if use_tensor_cores and torch.cuda.is_available():
        env['PYTORCH_CUDA_TF32'] = '1'  # Enable TF32 for faster training on Ampere+
        env['CUDA_LAUNCH_BLOCKING'] = '0'
    
    try:
        import subprocess
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        
        output_lines = []
        for line in process.stdout:
            if training_progress.should_stop:
                process.terminate()
                break
            output_lines.append(line)
            print(line, end='')
        
        training_progress.status = "completed" if process.returncode == 0 else "failed"
        return "".join(output_lines)
    except Exception as e:
        training_progress.status = "error"
        return f"Training error: {str(e)}"


def stop_training():
    """Stop training process"""
    global training_progress
    training_progress.stop()
    return "Training stopping..."


def create_ui():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Real-ESRGAN Web UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Real-ESRGAN Training & Inference Web Interface")
        gr.Markdown("Train and run super-resolution models with optimized performance including Tensor Core support")
        
        # Hardware Info Tab
        with gr.Tab("📊 Hardware Info"):
            hw_info = gr.Textbox(label="Hardware Information", value=get_device_info(), lines=10, interactive=False)
            refresh_hw_btn = gr.Button("Refresh Hardware Info")
            refresh_hw_btn.click(fn=get_device_info, outputs=hw_info)
        
        # Inference Tab
        with gr.Tab("🔍 Inference"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Settings")
                    input_image = gr.Image(type="pil", label="Input Image")
                    model_dropdown = gr.Dropdown(
                        choices=[
                            'RealESRGAN_x4plus',
                            'RealESRNet_x4plus', 
                            'RealESRGAN_x4plus_anime_6B',
                            'RealESRGAN_x2plus',
                            'realesr-animevideov3',
                            'realesr-general-x4v3'
                        ],
                        value='RealESRGAN_x4plus',
                        label='Model'
                    )
                    denoise_slider = gr.Slider(0, 1, value=0.5, step=0.1, label='Denoise Strength (for generalesr-x4v3)')
                    outscale_slider = gr.Slider(1, 8, value=4, step=0.5, label='Output Scale')
                    tile_slider = gr.Slider(0, 512, value=256, step=32, label='Tile Size (0 for no tiling)')
                    fp16_checkbox = gr.Checkbox(value=True, label='Use FP16 (Half Precision)')
                    enhance_btn = gr.Button("🚀 Enhance Image", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Output")
                    output_image = gr.Image(type="pil", label="Enhanced Image")
                    status_text = gr.Textbox(label="Status", interactive=False)
            
            enhance_btn.click(
                fn=enhance_image,
                inputs=[input_image, model_dropdown, denoise_slider, fp16_checkbox, tile_slider, outscale_slider],
                outputs=[output_image, status_text]
            )
            
            # Batch Processing
            gr.Markdown("### Batch Processing")
            with gr.Row():
                with gr.Column():
                    batch_input = gr.Textbox(label="Input Folder Path", placeholder="/path/to/input/folder")
                    batch_output = gr.Textbox(label="Output Folder Path", placeholder="/path/to/output/folder")
                    batch_btn = gr.Button("📁 Process Batch", variant="secondary")
                    batch_status = gr.Textbox(label="Batch Status", interactive=False)
            
            batch_btn.click(
                fn=enhance_batch,
                inputs=[batch_input, batch_output, model_dropdown, denoise_slider, fp16_checkbox, tile_slider, outscale_slider],
                outputs=batch_status
            )
        
        # Training Tab
        with gr.Tab("🎯 Training"):
            gr.Markdown("### Training Configuration")
            with gr.Row():
                with gr.Column():
                    config_file = gr.Textbox(
                        label="Config File Path",
                        value="options/train_realesrgan_x4plus.yml",
                        placeholder="options/train_realesrgan_x4plus.yml"
                    )
                    resume_state = gr.Textbox(
                        label="Resume State (optional)",
                        placeholder="Path to .state file to resume training"
                    )
                    tensor_cores = gr.Checkbox(
                        value=True,
                        label="Enable Tensor Core Optimization (TF32/FP16)"
                    )
                    num_gpu = gr.Slider(1, 8, value=1, step=1, label="Number of GPUs")
                    train_btn = gr.Button("🔥 Start Training", variant="primary")
                    stop_btn = gr.Button("⏹ Stop Training", variant="stop")
                
                with gr.Column():
                    training_output = gr.Textbox(label="Training Log", lines=20, interactive=False)
            
            train_btn.click(
                fn=start_training,
                inputs=[config_file, resume_state, tensor_cores, num_gpu],
                outputs=training_output
            )
            stop_btn.click(fn=stop_training, outputs=None)
        
        # Help Tab
        with gr.Tab("❓ Help"):
            gr.Markdown("""
            ## How to Use
            
            ### Inference
            1. Upload an image or provide a path to a folder
            2. Select the model you want to use
            3. Adjust settings as needed
            4. Click "Enhance Image" or "Process Batch"
            
            ### Training
            1. Configure your training options in a YAML file
            2. Point to the config file
            3. Enable Tensor Core optimization for faster training on RTX GPUs
            4. Start training
            
            ### Performance Tips
            - **Tensor Cores**: Enable for 2-3x speedup on RTX 20xx/30xx/40xx series
            - **FP16**: Use half precision for faster inference with minimal quality loss
            - **Tile Size**: Reduce if you encounter OOM errors, increase for better performance
            - **Batch Size**: Adjust based on your GPU memory
            
            ### Supported Hardware
            - **NVIDIA GPUs**: Full support with CUDA and Tensor Cores
            - **RT Cores**: Not directly used (they're for ray tracing), but Tensor Cores accelerate matrix operations
            """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN Web UI')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--share', action='store_true', help='Create public link')
    args = parser.parse_args()
    
    demo = create_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()

# This script is used to firstly generate an image with a separate prompt (i.e. a background image) and then inpaint it with a regular pipeline

import modules.scripts as scripts
import gradio as gr

from modules import processing
from modules.processing import Processed, StableDiffusionProcessingTxt2Img
from modules.shared import sd_model

class Script(scripts.Script):
    def title(self):
        return "Inpainting pre-generation"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        pregen_prompt = gr.Textbox(label="Pre-generation prompt", lines=2, interactive=True, value = "")
        pregen_negative_prompt = gr.Textbox(label="Pre-generation negative prompt", lines=2, interactive=True, value = "")

        return [pregen_prompt, pregen_negative_prompt]
    
    def run(self, p, pregen_prompt, pregen_negative_prompt):

        print('Inpainting pre-generation: creating init images from a text prompt')
        p_txt = StableDiffusionProcessingTxt2Img(
                sd_model=sd_model,
                outpath_samples=p.outpath_samples,
                outpath_grids=p.outpath_samples,
                prompt=pregen_prompt,
                styles=p.styles,
                negative_prompt=pregen_negative_prompt,
                seed=p.seed,
                subseed=p.subseed,
                subseed_strength=p.subseed_strength,
                seed_resize_from_h=p.seed_resize_from_h,
                seed_resize_from_w=p.seed_resize_from_w,
                sampler_name=p.sampler_name,
                batch_size=p.batch_size,
                n_iter=p.n_iter,
                steps=p.steps,
                cfg_scale=p.cfg_scale,
                width=p.width,
                height=p.height,
                restore_faces=p.restore_faces,
                tiling=p.tiling,
                enable_hr=None,
                denoising_strength=None,
            )
        
        processed = processing.process_images(p_txt)
        p_pics = processed.images
        p_info = processed.info
        initial_seed = processed.seed

        # splitting the pics into batches
        pics_batches = [p_pics[i:i+p.batch_size] for i in range(0, len(p_pics), p.batch_size)]

        print('Inpainting pre-generation: actually inpainting the generated images')
        p.n_iter = 1
        output_images = []

        for pics in pics_batches:
            p.init_images = pics
            processed = processing.process_images(p)
            output_images += processed.images
            p_info + '\n' + processed.info
        processed = Processed(p, output_images, initial_seed, p_info)
        return processed

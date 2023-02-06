# This script is used to firstly generate an image with a separate prompt (i.e. a background image) and then inpaint it with a regular pipeline

import modules.scripts as scripts
import gradio as gr

from modules import processing
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, create_infotext
from modules.shared import sd_model, opts
import modules.images as images

class Script(scripts.Script):
    def title(self):
        return "Inpainting pre-generation"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        pregen_prompt = gr.Textbox(label="Pre-generation prompt", lines=2, interactive=True, value = "")
        pregen_negative_prompt = gr.Textbox(label="Pre-generation negative prompt", lines=2, interactive=True, value = "")

        override_steps_count = gr.Checkbox(label="Override pregen steps count", value=False, interactive=True)
        pregen_steps_count = gr.Slider(minimum=1, maximum=150, step=1, label='Pre-generation steps count', value=20, interactive=True)

        return [pregen_prompt, pregen_negative_prompt, override_steps_count, pregen_steps_count]
    
    def run(self, p, pregen_prompt, pregen_negative_prompt, override_steps_count, pregen_steps_count):
        do_not_save_grid = p.do_not_save_grid
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
                steps=p.steps if not override_steps_count else pregen_steps_count,
                cfg_scale=p.cfg_scale,
                width=p.width,
                height=p.height,
                restore_faces=p.restore_faces,
                tiling=p.tiling,
                enable_hr=None,
                denoising_strength=None,
                do_not_save_grid=True
            )
        
        processed = processing.process_images(p_txt)
        p_pics = processed.images
        p_info = processed.info
        initial_seed = processed.seed

        # splitting the pics into batches
        pics_batches = [p_pics[i:i+p.batch_size] for i in range(0, len(p_pics), p.batch_size)]

        print('Inpainting pre-generation: actually inpainting the generated images')
        p.n_iter = 1
        p.do_not_save_grid=True
        output_images = []

        for pics in pics_batches:
            p.init_images = pics
            processed = processing.process_images(p)
            output_images += processed.images
            p_info = p_info + '\n' + processed.info
        
        index_of_first_image = 0
        p.do_not_save_grid = do_not_save_grid
        if not do_not_save_grid:
            index_of_first_image = to_grid(p, p_info, "grid", output_images)
        
        processed = Processed(p, output_images, initial_seed, p_info, index_of_first_image=index_of_first_image)
        return processed
    
def to_grid(p, p_info, grid_type, output_images):
    index_of_first_image = 0
    unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
    if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
        grid = images.image_grid(output_images, p.batch_size)

        if opts.return_grid:
            if opts.enable_pnginfo:
                grid.info["parameters"] = p_info
            output_images.insert(0, grid)
            index_of_first_image = 1

        if opts.grid_save:
            images.save_image(grid, p.outpath_grids, grid_type, p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=p_info, short_filename=not opts.grid_extended_filename, p=p, grid=True)
    return index_of_first_image

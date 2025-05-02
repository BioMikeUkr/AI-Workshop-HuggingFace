import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
from model import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = model.config.latent_dim

def generate_from_noise():
    z = torch.randn(1, latent_dim).to(device)
    with torch.no_grad():
        generated = model.decode(z)
    gen_img = generated.squeeze(0).permute(1, 2, 0).cpu().numpy()
    gen_pil = Image.fromarray((gen_img * 255).astype("uint8")).resize((512, 512))
    return gen_pil

def get_interface():
    with gr.Blocks() as iface:
        gr.Markdown("## Generate from Random Noise")
        generate_button = gr.Button("Generate Image")
        output_image = gr.Image(label="Generated Image", type="pil")
        generate_button.click(fn=generate_from_noise, inputs=[], outputs=output_image)

        examples = [[]]

        gr.Examples(
            examples=examples,
            inputs=[],
            outputs=output_image,
            fn=generate_from_noise,
            cache_examples=False
        )
    return iface

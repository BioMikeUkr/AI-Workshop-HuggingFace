import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
import numpy as np
from model import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize_input = transforms.Resize((32, 32))
to_tensor = transforms.ToTensor()

def reconstruct_image(image):
    image = Image.fromarray(image).convert('RGB')
    image_32 = resize_input(image)
    image_tensor = to_tensor(image_32).unsqueeze(0).to(device)
    with torch.no_grad():
        mu, _ = model.encode(image_tensor)
        recon = model.decode(mu)
    recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_img = Image.fromarray((recon_np * 255).astype(np.uint8)).resize((512, 512))
    orig_resized = image_32.resize((512, 512))
    return orig_resized, recon_img

def get_interface():
    with gr.Blocks() as iface:
        gr.Markdown("## Encoding & Reconstruction")
        with gr.Row():
            input_image = gr.Image(label="Input (Downsampled to 32x32)", type="numpy")
            output_image = gr.Image(label="Reconstructed", type="pil")
        run_button = gr.Button("Run Reconstruction")

        run_button.click(fn=reconstruct_image, inputs=input_image, outputs=[input_image, output_image])

        examples = [
            ["example_images/image1.jpg"],
            ["example_images/image2.jpg"],
            ["example_images/image3.jpg"],
            ["example_images/image10.jpg"],
            ["example_images/image4.jpg"],
            ["example_images/image5.jpg"],
            ["example_images/image6.jpg"],
            ["example_images/image7.jpg"],
            ["example_images/image8.jpg"],
        ]

        gr.Examples(
            examples=examples,
            inputs=[input_image],
        )
    return iface

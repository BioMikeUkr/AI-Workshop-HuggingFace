import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
import numpy as np
from model import model
import tempfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

resize_output = transforms.Resize((512, 512))

def interpolate_vectors(v1, v2, num_steps):
    return [v1 * (1 - alpha) + v2 * alpha for alpha in np.linspace(0, 1, num_steps)]

def to_pil(img_tensor):
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)

def interpolate_images_gif(img1, img2, num_interpolations=10, duration=100):
    img1 = Image.fromarray(img1).convert('RGB')
    img2 = Image.fromarray(img2).convert('RGB')
    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        mu1, _ = model.encode(img1_tensor)
        mu2, _ = model.encode(img2_tensor)
        interpolated = interpolate_vectors(mu1, mu2, num_interpolations)
        decoded_images = []
        for z in interpolated:
            out = model.decode(z)
            img = to_pil(out)
            img_resized = resize_output(img)
            decoded_images.append(img_resized)

    tmp_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    decoded_images[0].save(
        tmp_file.name,
        save_all=True,
        append_images=decoded_images[1:],
        duration=duration,
        loop=0
    )
    return tmp_file.name

def get_interface():
    with gr.Blocks() as iface:
        gr.Markdown("## Latent Space Interpolation (GIF Output)")
        with gr.Row():
            img1 = gr.Image(label="First Image", type="numpy")
            img2 = gr.Image(label="Second Image", type="numpy")
        slider_steps = gr.Slider(5, 30, value=10, step=1, label="Number of Interpolations")
        slider_duration = gr.Slider(50, 500, value=100, step=10, label="Duration per Frame (ms)")
        output_gif = gr.Image(label="Interpolation GIF")
        run_button = gr.Button("Interpolate")

        run_button.click(
            fn=interpolate_images_gif,
            inputs=[img1, img2, slider_steps, slider_duration],
            outputs=output_gif
        )

        examples = [
            ["example_images/image1.jpg", "example_images/image2.jpg", 10, 100],
            ["example_images/image3.jpg", "example_images/image4.jpg", 15, 150],
            ["example_images/image5.jpg", "example_images/image6.jpg", 20, 200],
            ["example_images/image7.jpg", "example_images/image8.jpg", 25, 250],
        ]

        gr.Examples(
            examples=examples,
            inputs=[img1, img2, slider_steps, slider_duration],
            outputs=output_gif,
            fn=interpolate_images_gif,
            cache_examples=False
        )
    return iface

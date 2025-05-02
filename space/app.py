import gradio as gr
from encoding import get_interface as encoding_page
from generation import get_interface as generation_page
from interpolation import get_interface as interpolation_page

with gr.Blocks() as demo:
    with gr.Tab("Encode & Reconstruct"):
        encoding_page()
    with gr.Tab("Generate from Noise"):
        generation_page()
    with gr.Tab("Interpolate"):
        interpolation_page()

demo.launch()

# AI-Workshop-HuggingFace-as-a-platform-for-AI-services
# Emoji VAE Workshop Project

This repository contains all the code and resources used for the **AI Workshop** session on **Hugging Face as a Platform for AI Services**, presented by *Mykhailo Shtopko*.

## 🔍 Project overview

In this project:

- Built a **Variational Autoencoder (VAE)** for emoji data and integrate it with the Hugging Face,
- Prepared a **custom emoji dataset** and uploaded it to Hugging Face Hub,
- Developed a **Gradio Space** with:
    - **Encoding & Reconstruction**: Upload an image and see its reconstruction,
    - **Generation from Noise**: Sample new images from latent space,
    - **Interpolation**: Create smooth morphing GIFs between two images.

## 🗂️ Project structure

- `/space/`: Gradio Space implementation (encoding, generation, interpolation)
- `emoji_vae.ipynb`: Notebook for training & testing the VAE
- `emoji_dataset_processing.ipynb`: Dataset preparation & Hugging Face upload script
s
## Quickstart
1. Firstly, clone this repo:
```comandline
git clone https://github.com/BioMikeUkr/AI-Workshop-HuggingFace.git
```
2. Preparing the dataset https://colab.research.google.com/github/BioMikeUkr/AI-Workshop-HuggingFace/blob/main/emoji_dataset_processing.ipynb

3. Building the model https://colab.research.google.com/github/BioMikeUkr/AI-Workshop-HuggingFace/blob/main/emoji_vae.ipynb

4. To create the space, first create a new Space on Hugging Face, then drag and drop the contents of the space/ directory into the upload interface.

## 🔗 Hugging Face links

- **Model:** [BioMike/emoji_vae](https://huggingface.co/BioMike/emoji-vae-init)
- **Dataset:** [BioMike/emoji-vae-dataset](https://huggingface.co/datasets/BioMike/emoji-vae-dataset)
- **Space:** [EmojiVAE](https://huggingface.co/spaces/BioMike/EmojiVAE)

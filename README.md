# Image Captioning with Transformers

This repository contains a Jupyter Notebook that implements an image captioning model using the CPTR architecture, as described in the research paper "[CPTR: Full Transformer Network for Image Captioning
](https://arxiv.org/pdf/2101.10804)" by Liu et al. (2021).

<img src='https://media.licdn.com/dms/image/C4D12AQGA3qFX3peTbw/article-cover_image-shrink_720_1280/0/1648387317335?e=2147483647&v=beta&t=4VOpEV8ptM4B4Q0UTZJUWqv4QFQvIuCubBoQLzJazds' width='800'>

## Project Overview

The goal of this project is to build a model that can generate descriptive captions for images. The CPTR architecture leverages the power of Vision Transformers (ViT) for image encoding and a Transformer decoder for text generation.

## Training and Evaluation

- **Training:**
   - The model is trained using the Adam optimizer with an initial learning rate of 3e-5.
   - A multiplicative learning rate scheduler is employed to adjust the learning rate during training for the last 10 epochs.
   - Hyperparameters:
     - **Epochs**: 20
     - **Batch Size**: 40
     - **Encoder Layers**: 12
     - **Decoder Layers**: 4
     - **Number of Heads**: 12
     - **Embed Dimentions**: 768
   - Masked cross-entropy Loss is used as the loss function we minimize.
   - We use a masked accuracy function to evaluate how well the model generates the highest probability tokens during training.

- **Evaluation:**
   - Beam search is used during inference to generate multiple candidate captions and select the one with the highest probability.
   - We set beam size to 3.

# Results on Unseen Images

<img src="https://github.com/danplotkin/PyTorchImageCaptionerTransformer/blob/main/results/result1.png">
<img src="https://github.com/danplotkin/PyTorchImageCaptionerTransformer/blob/main/results/result2.png">
<img src="https://github.com/danplotkin/PyTorchImageCaptionerTransformer/blob/main/results/result3.png">

## Key Features

- **CPTR Architecture:** The model follows the CPTR architecture, which consists of:
   - **ViT Image Encoder:** A pre-trained Vision Transformer (ViT) extracts feature representations from the input image.
   - **Word Embeddings**: I have a custom built word embedding layer with sinusoidal positional encoding.
   - **Transformer Decoder:** A Transformer decoder generates the caption word by word, attending to both the image features and the previously generated words.
   - **Fully Connected Layer:** The fully connected layer maps our decoder output to a shape of (batch_size, seq_len, vocab_size).

- **Custom Dataset:** Implements a `Flickr8kDataset` class for efficient data loading and preprocessing.
- **Training and Validation:** Provides code for building a custom trainer to use teacher forcing during training. We then use autoregressive decoding during evaluation, using both greedy and beam search methods.
  
## Getting Started
1. **Download the Flickr8k Dataset:**
   - Obtain the dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place it in the project directory.

2. **Set Up Google Colab (Optional):**
   - If using Google Colab, mount your Google Drive and store your Kaggle credentials using `google.colab.userdata`.

3. **Run the Jupyter Notebook:**
   - Open and execute the `image_captioning.ipynb` notebook to train and evaluate the model.
  
## Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danplotkin/PyTorchImageCaptionerTransformer/blob/main/ImageCaptionerPytorch.ipynb)

## Acknowledgments

- The CPTR architecture is based on the work of Liu et al. (2021).
- The Flickr8k dataset is used for training and evaluation.


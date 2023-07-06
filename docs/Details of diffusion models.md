# Diffusion models

This note leverages the following resources:

- Probablistic machine learning - advanced topics, by Kevin Murphy

## Introduction

A diffusion model consists of an encoder and a decoder. The encoder is used for foward diffusion process, where noise is gradually added to the input of the encoder, until the input becomes completely noise. The encoder is predefined so it does not contain any trainable paramenters. The decoder tries to recover the original input from the noise. The decoding function needs to be trained and it contains trainable parameters. 

## Denoising diﬀusion probabilistic models (DDPMs)

DDPM can be easily trained without any risk of posterior collapse.

### Encoder (forwards diﬀusion)

Let us denote

- $q$ -- encoder
- $T$ -- maximun of timesteps
- $t$ -- index of timesteps
- $\boldsymbol{x}_t$ -- input at timestep $t$
- $q\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{0}\right)$ -- diffusion kernel which can be analytically computed

With the assumption of Markov property, we can have the encoding process as
$$
q\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{x}_{0}\right)=\prod_{t=1}^{T} q\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{t-1}\right)
$$
At the end of the forward diffusion process, $q\left(\boldsymbol{x}_{T} \mid \boldsymbol{x}_{0}\right)$ should be noise.

### Decoder (reverse diffusion)

Let us denote:

- $p_{\theta}$  -- decoder
- $\theta$  -- trainable parameters of the decoder. 

With the assumption of Markov property, we can have the decoding process as
$$
p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{0: T}\right) = p\left(\boldsymbol{x}_{T}\right) \prod_{t=1}^{T} p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}\right)
$$
where $p(\boldsymbol{x}_T)$ is the distribution of noise. At the end of the decoding process, 

### Model fitting

We fit the DDPM model by maximizing the evidence lower bound (ELBO). In particular, for each data example $x_0$ we have
$$
\begin{aligned} \log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{0}\right) & =\log \left[\int d \boldsymbol{x}_{1: T} q\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{x}_{0}\right) \frac{p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{0: T}\right)}{q\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{x}_{0}\right)}\right] \\ & \geq \int d \boldsymbol{x}_{1: T} q\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{x}_{0}\right) \log \left(p\left(\boldsymbol{x}_{T}\right) \prod_{t=1}^{T} \frac{p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}\right)}{q\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{t-1}\right)}\right) \\ & =\mathbb{E}_{q}\left[\log p\left(\boldsymbol{x}_{T}\right)+\sum_{t=1}^{T} \log \frac{p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}\right)}{q\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{t-1}\right)}\right] \end{aligned}
$$
After some algebra, we have the negative ELBO (variational upper bound) as
$$
\begin{aligned} \mathcal{L}\left(\boldsymbol{x}_{0}\right) & =-\mathbb{E}_{q\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{x}_{0}\right)}\left[\log \frac{p\left(\boldsymbol{x}_{T}\right)}{q\left(\boldsymbol{x}_{T} \mid \boldsymbol{x}_{0}\right)}+\sum_{t=2}^{T} \log \frac{p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}\right)}{q\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}, \boldsymbol{x}_{0}\right)}+\log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{1}\right)\right] \\ & =\underbrace{D_{\mathbb{K} \mathbb{L}}\left(q\left(\boldsymbol{x}_{T} \mid \boldsymbol{x}_{0}\right) \| p\left(\boldsymbol{x}_{T}\right)\right)}_{L_{T}\left(\boldsymbol{x}_{0}\right)} \\ & +\sum_{t=2}^{T} \mathbb{E}_{q\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{0}\right)} \underbrace{D_{\mathbb{K} \mathbb{L}}\left(q\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}, \boldsymbol{x}_{0}\right) \| p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}\right)\right)}_{L_{t-1}\left(\boldsymbol{x}_{0}\right)}-\underbrace{\mathbb{E}_{q\left(\boldsymbol{x}_{1} \mid \boldsymbol{x}_{0}\right)} \log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{1}\right)}_{L_{0}\left(\boldsymbol{x}_{0}\right)}\end{aligned}
$$
As we can see, $\mathcal{L}(\boldsymbol{x}_0)$ has three parts

- $L_{T}\left(\boldsymbol{x}_{0}\right)$ -- prior loss: similarity between $\boldsymbol{x}_T$ and noise
- $L_{t-1}\left(\boldsymbol{x}_{0}\right)$ -- diffusion loss: similarity between analytical distribution and estimated distribution of $\boldsymbol{x}_{t-1}$
- $L_{0}\left(\boldsymbol{x}_{0}\right)$ -- reconstruction loss: quality of estimated distribution of $x_0$

Then we can minimize $\mathcal{L}$ over the entire dataset. $\mathcal{L}$ is a differentiable function of trainable parameters $\theta$. Note that $L_T(\boldsymbol{x}_0)$ does not contains any trainable parameters, and it is usually igorened during training.

### Learning the noise schedule

The noise schedule is often linear or cosine. There are some ways of optimizing the noise schedule, such as variational diﬀusion model. We will discuss it later.

### Example: image generation

The most common architecture for image generation is based on the U-net model.

## Score-based generative models (SGMs)

TBA

## Continuous time models using diﬀerential equations

TBA

## Speeding up diﬀusion models

TBA

## Conditional generation

In this we discuss how to generate samples from a diﬀusion model where we condition on some side information, such as a class label or text prompt.

Let us denote

- $\boldsymbol{c}$ -- side information

### Conditional diffusion model

TBA

### Classifier guidance


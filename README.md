# Wellearn

## Introduction

This repository aims to provide a set of tools to help data scientists, machine learning engineers and researchers to build fair machine learning models.

## Installation

```bash
conda create -n wellearn python=3.9
conda activate wellearn
pip install -r requirements.txt
pip install -e .
```

## Datasets

### [MIMIC-III](https://mimic.mit.edu/docs/iii/)

MIMIC-III (**M**edical **I**nformation **M**art for **I**ntensive **C**are III) is a large, freely-available database comprising deidentified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.

The database includes information such as demographics, vital sign measurements made at the bedside (~1 data point per hour), laboratory test results, procedures, medications, caregiver notes, imaging reports, and mortality (both in and out of hospital).

MIMIC supports a diverse range of analytic studies spanning epidemiology, clinical decision-rule improvement, and electronic tool development. It is notable for three factors:

- it is freely available to researchers worldwide
- it encompasses a diverse and very large population of ICU patients
- it contains high temporal resolution data including lab results, electronic documentation, and bedside monitor trends and waveforms.

### Intersectional bias assessment for depression prediction

This synthetic dataset contains demographic and clinical data used to train a classifier to predict a diagnosis (of schizophrenia or depression). This dataset is used in the tutorial 'An Intersectional Approach to Model Construction and Evaluation in Mental Health Care' presented at ACM FAccT 2022 [link to the tutorial](https://facctconference.org/2022/acceptedtuts.html#construct). It can be download in OpenML [here](https://openml.org/search?type=data&status=active&id=45040).

## Resources

- [Awesome-Diffusion-Models](https://github.com/heejkoo/Awesome-Diffusion-Models)
  
  - Resouces about diffusion models

- [tab-ddpm](https://github.com/yandex-research/tab-ddpm)
  
  - Generate tabular data with DDPM

- [ehrdiff](https://github.com/sczzz3/ehrdiff)
  
  - Generate binary EHR data (MIMIC III) with DDPM
  - Have MIMIC III data

- [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
  
  - DDPM code

- Diffusion models from scratch in PyTorch
  
  - [YouTube](https://www.youtube.com/watch?v=a4Yfz2FxXiY)
  - [Colab](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=HhIgGq3za0yh)

## Coding style

We use Goolge Python Style Guide. Please refer to [this link](https://google.github.io/styleguide/pyguide.html) for more details.

## Acknowledgement

Mentors:

- Peikun Guo
- Khidijia Zanna
- Akane Sano
- ChatGPT

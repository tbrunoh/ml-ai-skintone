# Skin Tone Classifier

> A PyTorch model that classifies skin tone from a photo on the 10 point Monk Skin Tone scale.

**Part of [Inclusive Skin AI](https://github.com/tbrunoh#inclusive-skin-ai)**, an end to end system for skin analysis built for melanin-rich skin first. Most skin analysis models learn from lighter skin and misread darker tones; this project starts from the other end.

## What it does

Takes a face or skin photo and predicts its Monk Skin Tone (MST) class, from 1 (lightest) to 10 (darkest). The model is packaged as the `mlaist` Python library so the inference API and notebooks share one implementation.

## How it works

- **Model:** a five block convolutional network (32 to 512 channels, batch norm, ReLU) with global average pooling and a three layer classifier head, on 512 by 512 RGB input.
- **Loss:** ordinal regression. Each MST class is encoded as a cumulative binary target, so predicting 7 for a true 8 costs less than predicting 2. Skin tone is ordered, and the loss respects that.
- **Data:** 1,208 training and 302 test images, each labeled with an MST class (`mlaist/data/`). The CSVs reference image URLs in retired storage; point them at your own images to retrain.
- **Serving:** `inference.py` runs the trained weights locally; `cloud-inference.py` calls a Vertex AI endpoint.

## Quickstart

```bash
git clone https://github.com/tbrunoh/ml-ai-skintone.git
cd ml-ai-skintone
pip install -r Requirements.txt
# set dataset, checkpoint and model paths in .env, then:
python -m mlaist.train
```

Install as a library:

```bash
pip install git+https://github.com/tbrunoh/ml-ai-skintone.git
```

## Repository layout

```
mlaist/
  models/model.py        SkinToneCNNModel
  models/loss_func.py    OrdinalRegressionLoss
  models/dataset.py      CSV backed image dataset
  train.py               training loop with checkpoints
  inference.py           local inference
  cloud-inference.py     Vertex AI inference
  data/                  MST labeled train and test splits
  output/                trained weights
```

## The Inclusive Skin AI repositories

| Layer | Repository | Role |
|:--|:--|:--|
| MODEL | **ml-ai-skintone** | Skin tone classifier on the Monk Skin Tone scale |
| MODEL | [ml-ai-skin-conditions](https://github.com/tbrunoh/ml-ai-skin-conditions) | Skin condition model package, early stage |
| CORE | [ml-ai-core](https://github.com/tbrunoh/ml-ai-core) | Shared PyTorch base classes and utilities |
| SERVE | [ml-ai-client-api](https://github.com/tbrunoh/ml-ai-client-api) | FastAPI inference service for the skin tone model |
| DATA | [ml-data-upload-api](https://github.com/tbrunoh/ml-data-upload-api) | FastAPI upload service feeding storage and labeling |
| DATA | [ml-data-labelling-ui](https://github.com/tbrunoh/ml-data-labelling-ui) | Label Studio deployment for image annotation |
| PRODUCT | [ml-ai-skintone-ui](https://github.com/tbrunoh/ml-ai-skintone-ui) | Next.js skin tone analysis app |
| PRODUCT | [ml-landing-page](https://github.com/tbrunoh/ml-landing-page) | Next.js landing page for the consumer app |
| INFRA | [ml-terraform](https://github.com/tbrunoh/ml-terraform) | Azure infrastructure as code |
| INFRA | [ml-terraform-modules](https://github.com/tbrunoh/ml-terraform-modules) | Reusable Terraform modules |
| RESEARCH | [ml-ai-samples](https://github.com/tbrunoh/ml-ai-samples) | Reference PyTorch experiments |

## Status

The cloud environment behind this project has been retired. The code is published as a reference implementation, and every endpoint, key and project identifier in configuration files is a placeholder you replace with your own.

---

<sub>Maintained by <a href="https://github.com/tbrunoh">Tom Bruno</a>. Original engineering by Nelson Kigen. Part of <a href="https://github.com/tbrunoh#inclusive-skin-ai">Inclusive Skin AI</a>.</sub>

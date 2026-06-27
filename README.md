# Improved-zero-RRR

> Project Seminar: *Reproduce Research Results* — FAU Erlangen-Nürnberg

Reproduction of results from the paper **"Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions"** (CVPR 2024).

📄 Paper: [Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions](https://openaccess.thecvf.com/content/CVPR2024/html/Saha_Improved_ZeroShot_Classification_by_Adapting_VLMs_with_Text_Descriptions_CVPR_2024_paper.html)
💻 Original repo: [cvl-umass/AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)

---

## About this project

This repository documents a **reproduction** of the paper above, carried out as part of the *Project Seminar: Reproduce Research Results* at **FAU Erlangen-Nürnberg**. The seminar's premise is straightforward: take a published ML paper and independently reproduce its reported results, rather than conduct new research. The work here is a reproduction effort, not an original contribution.

The goals of this specific reproduction were to:
- **Practice the end-to-end workflow** of reproducing a deep learning paper — setting up the environment, acquiring the dataset, downloading provided checkpoints, and running the original evaluation code.
- **Verify the paper's reported numbers**, specifically the `CLIP_FT + A` accuracy in Table 1, by independently re-running inference on the CUB-200-2011 dataset.
- **Document the process and any friction points** encountered along the way, so the steps here can be a reference for similar reproduction work.

All credit for the underlying method, codebase, and pre-trained checkpoints belongs to the original authors and the maintainers of the [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS) repository (see [Acknowledgements](#acknowledgements)). This repo adds only the documentation, environment setup, and run instructions used to reproduce one of their reported results — it introduces no new method or claim of its own.

All steps below were run on **Google Colab**.

---

## 1. Clone the original repository

```bash
!git clone https://github.com/cvl-umass/AdaptCLIPZS.git
%cd AdaptCLIPZS
```

## 2. Install dependencies

The original repo ships an `environment.yml`, but conda-based environment files aren't compatible with Colab, so dependencies were installed manually instead:

```bash
!pip install torch torchvision timm transformers opendatasets matplotlib openai
!pip install -U scikit-image scikit-learn tqdm
!pip install faiss-cpu fvcore yacs
!pip install gdown
!pip install torchfile
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install numpy scipy matplotlib openai timm transformers opendatasets fvcore yacs
!pip install faiss-cpu clip-by-openai ftfy regex pillow tensorboard tensorboardx
!pip install requests click cython python-dateutil pandas packaging
!pip install absl-py addict altair antlr4-python3-runtime asttokens attrs backcall beautifulsoup4 bleach blis braceexpand catalogue cmake huggingface-hub langdetect markdown-it-py matplotlib-inline nltk numpy pandas pre-commit preshed protobuf spacy streamlit
```

## 3. Data preparation (CUB-200-2011)

Download the dataset (link provided in the original repo's README):

```bash
!wget -O CUB_200_2011.tgz "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
```

Extract all images into a single flat folder:

```bash
!tar -xvzf CUB_200_2011.tgz -C CUB_200_2011/
!mkdir -p CUB_200_2011/images_extracted
%cd CUB_200_2011/images/
!for folder in *; do mv "$folder"/* ../images_extracted/; done
%cd ../../../
```

## 4. Pre-trained checkpoints

Pre-trained checkpoints for iNaturalist21, NABirds, and CUB (both ViT-B/16 and ViT-B/32) are provided by the original authors:

```bash
!gdown --id 1YbAtXObtohm65ylJnSMfbo86xsHETs-H -O pretrained_models/CUB_b16.pth
```

Install the CLIP package:

```bash
!pip install git+https://github.com/openai/CLIP.git
```

## 5. Run inference

```bash
!python test_AdaptZS.py \
    --im_dir CUB_200_2011/images_extracted/ \
    --ckpt_path pretrained_models/CUB_b16.pth \
    --text_dir ./gpt_descriptions/gpt4_0613_api_CUB/ \
    --text_dir_loc ./gpt_descriptions/gpt4_0613_api_CUB_location/ \
    --arch ViT-B/16 \
    --attributes
```

## 6. Results

We reproduce the **CLIP_FT + A** value reported in Table 1 of the paper for the CUB dataset.

| Model                 | Paper-reported accuracy |
|-----------------------|:------------------------:|
| iNaturalist21_b32.pth | 54.54 |
| iNaturalist21_b16.pth | 56.76 |
| NABirds_b32.pth       | 55.46 |
| NABirds_b16.pth       | 56.59 |
| CUB_b32.pth           | 54.23 |
| **CUB_b16.pth**       | **56.01** |

**Our reproduced result (CUB_b16.pth):**

```
Accuracy Val = 55.529
```

This lands within ~0.5 points of the paper's reported 56.01 for the same checkpoint and architecture.

## 7. Reproduction notes

- The original `environment.yml` could not be used as-is in Colab, requiring a manual, package-by-package dependency install (see Step 2).
- Only the CUB-200-2011 / ViT-B/16 checkpoint was independently re-run for this seminar project. The iNaturalist21 and NABirds numbers in the table above are quoted directly from the paper for reference and were **not** re-run here.
- The small gap between the reproduced (55.53) and reported (56.01) accuracy is consistent with normal run-to-run variation rather than a methodological discrepancy, though no controlled re-runs were done to confirm this.

---

## Acknowledgements

This work reproduces results from:

> Saha, O. et al. *Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions.* CVPR 2024.

All credit for the method, model design, and original implementation goes to the paper's authors and the [cvl-umass/AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS) repository maintainers. This repository exists solely to document a reproduction of their reported results as part of coursework and makes no claim of original research contribution.

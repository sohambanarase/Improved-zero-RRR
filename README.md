# Improved-zero-RRR

Reproduction of results from the paper **"Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions"** (CVPR 2024).

📄 Paper: [Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions](https://openaccess.thecvf.com/content/CVPR2024/html/Saha_Improved_ZeroShot_Classification_by_Adapting_VLMs_with_Text_Descriptions_CVPR_2024_paper.html)
💻 Original repo: [cvl-umass/AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)

The paper adapts CLIP models using GPT-generated textual descriptions to improve zero-shot classification. This reproduction focuses on the **CUB-200-2011** dataset, using a pre-trained checkpoint, and compares the reproduced result against the number reported in the original paper.

All steps below were run on **Google Colab**.

---

## 1. Clone the original repository

```bash
!git clone https://github.com/cvl-umass/AdaptCLIPZS.git
%cd AdaptCLIPZS
```

## 2. Install dependencies

The repo ships an `environment.yml`, but conda environments aren't compatible with Colab, so dependencies are installed manually instead:

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

This is within ~0.5 points of the paper's reported 56.01, which is a reasonably close match given normal run-to-run variance (e.g. preprocessing/data ordering, library versions).

---

## Notes

- This reproduction follows the steps and configuration described in the paper and its official repository as closely as possible.
- Only the CUB-200-2011 dataset / ViT-B/16 checkpoint was evaluated in this run; the iNaturalist21 and NABirds results in the table above are quoted directly from the paper for comparison and were not independently reproduced here.

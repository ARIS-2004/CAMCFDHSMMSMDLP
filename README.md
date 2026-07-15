# CAMCFDHSMMSMDLP

A Context-Aware Multi-Class Framework for Detecting Hate Speech using Multi-Modality in Social Media: A Deep Learning Perspective.

This repository contains the source code, implementation notebooks, and supporting materials for the multi-modal hate speech detection framework proposed by Group 33, Techno Main Salt Lake, Kolkata.

---

## Authors

- Shaikh Aris Akhtar (Roll: 13000122003)
- Rimjhim Chowdhury (Roll: 13000122028)
- Suman Das (Roll: 13000123137)
- Ritom Mistry (Roll: 13000122128)

Supervisor: Prof. Poulami Dutta, Department of Computer Science and Engineering, Techno Main Salt Lake, Kolkata.

---

## Overview

The proposed framework performs six-class hate speech classification on the MMHS150K dataset by jointly modelling three complementary modalities:

1. Visual features extracted using a partially fine-tuned Vision Transformer (ViT-B/16).
2. Textual features extracted using a Custom Text Transformer trained from scratch.
3. Behavioural features captured through a self-supervised Social Encoder that regresses Shannon reaction entropy on 185,766 Facebook posts.

Features from the three modalities are fused using a Cross-Modal Attention mechanism and classified using four classifier heads (MLP, Transformer, 1D-CNN, BiLSTM). Class-Weighted Cross-Entropy Loss is used to mitigate the severe imbalance of the MMHS150K dataset.

---

## Repository Structure

```
CAMCFDHSMMSMDLP/
├── README.md                                    (this file)
├── requirements.txt                             (Python dependencies)
├── .gitignore                                   (files excluded from git)
└── notebooks/
    ├── Socialencoder.ipynb                      (Social Encoder training)
    ├── mutilmodal.ipynb                         (WithSE full pipeline)
    └── multimodalwithoutsocial_encoder.ipynb   (NoSE ablation)
```

---

## Datasets

Datasets are hosted separately on Google Drive due to size constraints (approximately 20 GB total).

**MMHS150K dataset (primary benchmark):**
Approximately 150,000 memes with image, overlay text, and tweet caption, annotated across six hate speech categories.
Download: https://drive.google.com/drive/folders/1CVrrdnpoDyFcWKJnX0Dfrr-MQb2fyjSP?usp=drive_link

**Facebook Memes dataset (auxiliary, for Social Encoder training):**
185,766 posts filtered from an original pool of 235,880 posts, containing multi-modal posts with seven categorical reactions (Like, Love, Haha, Wow, Sad, Angry, Care).
Download: https://drive.google.com/drive/folders/1MZhKOrmle-0GTv34tI7jCrHb-eIEuSLO?usp=drive_link

---

## Reproduction Steps

1. Clone this repository.

   ```
   git clone https://github.com/ARIS-2004/CAMCFDHSMMSMDLP.git
   cd CAMCFDHSMMSMDLP
   ```

2. Install dependencies.

   ```
   pip install -r requirements.txt
   ```

3. Download the datasets from the Google Drive links above. Place the MMHS150K dataset and the Facebook Memes dataset under a local folder named `data/`.

4. Run the notebooks in the following order:

   a. `notebooks/Socialencoder.ipynb` - trains the Social Encoder on the Facebook Memes dataset.

   b. `notebooks/mutilmodal.ipynb` - trains the full multi-modal pipeline (WithSE) on the MMHS150K dataset.

   c. `notebooks/multimodalwithoutsocial_encoder.ipynb` - trains the ablation configuration (NoSE) on the MMHS150K dataset.

---

## Experimental Setup

- Dataset: MMHS150K (149,823 memes, six classes: NotHate, Racist, Sexist, Homophobe, Religion, OtherHate).
- Splits tested: 80/10/10, 75/15/10, 70/20/10.
- Loss function: Class-Weighted Cross-Entropy with weights w_c = 1/f_c^0.3.
- Optimiser: AdamW.
- Hardware: NVIDIA GPU with CUDA 12.x.

---

## Reported Results

Best six-class Macro F1 obtained by the proposed framework:

- 80:10:10 split: MLP classifier, Macro F1 = 0.4391
- 75:15:10 split: MLP classifier, Macro F1 = 0.4464
- 70:20:10 split: Transformer classifier, Macro F1 = 0.4467

Ablation study confirms consistent improvement of the Social Encoder over the no-Social-Encoder baseline across all three splits.

---

## Citation

If this work is referenced in academic writing, please cite the associated paper (details will be provided upon publication).

---

## Contact

For questions related to this repository, please contact:

- Shaikh Aris Akhtar - shaikharisakhtar1234@gmail.com

---

## License

This repository is provided for academic and research purposes.

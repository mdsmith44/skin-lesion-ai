I'm trying to build my computer vision skills to prepare for a job that applies computer vision to healthcare industry.  The job also asks for experience with the NVIDIA tools such as NeMo and Nemotron.  Is there a sample project I can build that would help build my skills and familiarity in this area?  The skin cancer dataset at https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 seems interesting to me.


Yes. HAM10000 is an excellent dataset for this goal, particularly because you can turn it into a project that goes well beyond “train a CNN to classify skin lesions.” The strongest portfolio project would combine computer vision, medical-AI evaluation, explainability, and a vision-language model, while giving you a reason to work with NVIDIA's current NeMo/Nemotron ecosystem.
HAM10000 contains 10,015 dermatoscopic images covering seven diagnostic categories, with ground truth established through pathology, follow-up, expert consensus, or confocal microscopy. Nature
A project I'd recommend
"Multimodal Skin Lesion Analysis with Computer Vision and NVIDIA NeMo"
Build an end-to-end prototype that takes a dermatoscopic image and produces something like:
Predicted category: Melanoma
Confidence: 87%
Visual evidence: Irregular pigmentation and asymmetric lesion structure
Model explanation: ...
Important caveat: This is a research/educational model and is not a clinical diagnosis.

I'd structure it as five progressively more sophisticated stages.
## 1. Build a conventional computer-vision baseline
Start with HAM10000 and build a proper image-classification pipeline.
For example:
Input
dermatoscopy image → preprocessing → neural network → 7-class prediction
Try progressively more sophisticated models:
1. Simple CNN
2. ResNet/EfficientNet
3. Vision Transformer
4. Transfer learning from a pretrained model

Don't just report accuracy. Because HAM10000 is highly imbalanced, I'd evaluate:
- balanced accuracy
- macro F1
- per-class precision/recall
- confusion matrix
- ROC-AUC where appropriate
- sensitivity for melanoma
- specificity for melanoma
This alone gives you a legitimate computer-vision project.
But I'd deliberately not stop here.

## 2. Add medical-imaging-specific explainability
This is where the project starts becoming much more interesting for a healthcare-CV job.
Add Grad-CAM or another saliency technique and visualize:
Original image
→ model prediction
→ heatmap showing where the model was looking
For example:


             HAM10000 image
                    │
                    ▼
             Vision model
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Classification       Grad-CAM
          │                   │
          ▼                   ▼
       Melanoma          Heatmap
          │                   │
          └─────────┬─────────┘
                    ▼
             Model report

Then investigate an important healthcare-AI question:
Is the model looking at the lesion, or is it exploiting artifacts/background features?

That is a much more impressive question to investigate than simply maximizing accuracy.
You could deliberately test:
- images with borders/cropping
- different image backgrounds
- image artifacts
- hair
- lesion size
- acquisition source
HAM10000 is particularly suitable for this because its images came from different populations, modalities, and two clinical sites. Nature

## 3. Add a Vision-Language Model
This is the part I'd use to build your NeMo/Nemotron familiarity.
Instead of having the system output only:
melanoma: 0.87

have it produce a structured interpretation:
Diagnosis prediction:
Melanoma

Confidence:
0.87

Visual observations:
- Asymmetric lesion
- Irregular pigmentation
- Irregular border

Potentially concerning characteristics:
- Asymmetry
- Color variation

Important limitations:
The model is trained on dermatoscopic images and
should not be used as a clinical diagnostic system.
You can use a VLM to generate the natural-language portion while keeping the conventional CV model responsible for the actual classification.
That's particularly relevant to NVIDIA's current tooling. NeMo Framework now encompasses multimodal and computer-vision workloads, in addition to language models. NVIDIA Docs
And current NeMo AutoModel supports VLM training/fine-tuning, including image-text-to-text models. NVIDIA Docs.

## 4. Create your own multimodal training dataset
This could be the most educational part of the project.
HAM10000 doesn't naturally give you the sort of question/answer data needed to train a VLM. So construct a derived dataset.

For example:
{
  "image": "ISIC_0027419.jpg",
  "question": "What type of lesion is shown?",
  "answer": "Melanocytic nevus"
}
Then expand it:
{
  "image": "ISIC_0027419.jpg",
  "question": "What visual characteristics are present?",
  "answer": "The lesion is relatively symmetric with fairly uniform pigmentation."
}
And:
{
  "image": "ISIC_0027419.jpg",
  "question": "What diagnostic category does this lesion belong to?",
  "answer": "Melanocytic nevus."
}
You could generate multiple question types from the HAM10000 metadata and your CV model.
This would give you experience with an important real-world problem:
How do you turn an existing computer-vision dataset into a multimodal dataset suitable for foundation-model fine-tuning?

That's directly relevant to current NeMo workflows. NVIDIA's documentation specifically describes integrating custom image + text datasets using preprocessing and YAML configuration. NVIDIA Docs

## 5. Fine-tune a VLM with NeMo
Then bring NVIDIA explicitly into the project.
The workflow becomes:
                    
                    HAM10000
                       │
                       ▼
              Data preparation
                       │
                       ▼
          ┌──────────────────────┐
          │ Image + text dataset │
          └──────────┬───────────┘
                     │
                     ▼
             NVIDIA NeMo
                     │
                     ▼
              VLM fine-tuning
                     │
                     ▼
              Image + question
                     │
                     ▼
                 VLM answer

NeMo AutoModel currently supports PEFT/LoRA fine-tuning for supported VLMs, which makes this considerably more approachable than trying to train a large model from scratch. NVIDIA Docs
And NVIDIA's current documentation actually has an end-to-end example of fine-tuning a VLM on a medical VQA dataset using NeMo AutoModel. NVIDIA Docs
That's a particularly useful reference for your project because the NVIDIA example uses MedPix-VQA, a medical image/question/answer dataset.
Where Nemotron fits
I would not force Nemotron into the first version just because the job description mentions it.
Instead, make it the next stage.
For example:

Stage A — CV model

Image → ResNet/ViT → diagnosis

Stage B — VLM

Image + question → VLM → answer

Stage C — Nemotron-powered medical assistant

                  Image
                    │
                    ▼
             Vision model
                    │
             ┌──────┴──────┐
             │             │
       classification   observations
             │             │
             └──────┬──────┘
                    ▼
               Nemotron
                    │
                    ▼
          Structured explanation

The language model could take the structured outputs from your CV system and turn them into a coherent report.
That gives you an opportunity to demonstrate that you understand an important distinction:
A language model isn't necessarily the right model for the image classification task.
You can use specialized CV models for visual perception and an LLM/VLM for reasoning, interaction, and report generation.
That's much closer to how I'd expect a sophisticated healthcare AI system to be architected.
One particularly interesting experiment
I'd make this a central research question:
Can a VLM identify clinically relevant visual features without being explicitly told what to look for?
For example, give it an image and ask:
"Describe the visual characteristics of this lesion."

Then compare its response against:
- the actual HAM10000 diagnosis
- metadata
- your CV model's prediction
- Grad-CAM
- perhaps dermatologist-created descriptions from another dataset
You could investigate whether the VLM's explanations correlate with the features that actually drive the classifier.
That's a genuinely interesting CV/AI experiment rather than just a tutorial.
Your final portfolio project could look like this
I'd aim for a GitHub repository something like:

```
ham10000-nvidia-medical-ai/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_gradcam_analysis.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   ├── explainability/
│   └── vlm/
│
├── nemo/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── train.sh
│
├── app/
│   └── demo.py
│
├── README.md
└── results/
```

And ultimately have a small web application:

```
┌───────────────────────────────────────────────┐
│       Skin Lesion AI Research Demo            │
├───────────────────────────────────────────────┤
│                                               │
│        [ Upload dermatoscopic image ]         │
│                                               │
├───────────────────────────────────────────────┤
│ Prediction                                    │
│                                               │
│ Melanoma                         87%          │
│                                               │
│ ┌──────────────┐  ┌────────────────────────┐ │
│ │              │  │ Model explanation      │ │
│ │   IMAGE      │  │                        │ │
│ │              │  │ Asymmetric structure   │ │
│ │              │  │ and heterogeneous      │ │
│ │              │  │ pigmentation...        │ │
│ └──────────────┘  └────────────────────────┘ │
│                                               │
│ [Show Grad-CAM] [Ask VLM] [Compare Models]   │
└───────────────────────────────────────────────┘
```

That would give you something demonstrable in an interview, rather than simply saying "I trained a model on HAM10000."
How I'd sequence your learning
Given your goal, I would not start with NeMo.
I'd deliberately build the project in this order:

| Phase | Skill you develop | Technology |
|---|---|---|
| 1 | Medical image preprocessing | Python/OpenCV/PyTorch |
| 2 | CNN image classification | PyTorch |
| 3 | Transfer learning | ResNet/ViT |
| 4 | Medical ML evaluation | sklearn/PyTorch |
| 5 | Explainable CV | Grad-CAM |
| 6 | Vision-language models | Transformers |
| 7 | Multimodal dataset construction | JSON/YAML/Python |
| 8 | VLM fine-tuning | **NeMo AutoModel** |
| 9 | NVIDIA model ecosystem | **Nemotron** |
| 10 | Deployment | NIM/Triton/etc. |



This progression is useful because each stage teaches you something fundamental rather than hiding the computer-vision concepts behind a foundation model.
NVIDIA's current NeMo ecosystem also includes tools for deploying models and NIM provides managed/self-hosted inference interfaces for VLMs, so deployment can eventually become another part of the project. NVIDIA Docs
One important caveat
I'd describe the project as medical-AI research/education, not as a skin-cancer diagnostic system. HAM10000 is a valuable benchmark, but its labels and collection process have limitations, and performance on this dataset wouldn't establish clinical validity. The dataset authors themselves describe it as a training set for academic machine learning purposes. Nature
I think this could be an unusually good project for the job you're targeting because it lets you demonstrate three different capabilities in one coherent application:
computer vision → medical AI → NVIDIA generative/multimodal AI.
If you'd like, I can take this one step further and lay out a concrete 6–8 week project plan, including the specific PyTorch models, NVIDIA/NeMo/Nemotron models I'd use, datasets, milestones, and what you'd want to have in the GitHub repo at the end.
<p align="center">
<img src="./logo.png" width="500" height="180" />
</p>

# H-optimus-0
An open-source foundation model for histology.

## Dataset and model training.
### Dataset.
Our model was trained on a proprietary collection of more than 500,000 H&E stained whole slide histology images from which we extracted several hundreds of millions of tiles. 
Our dataset covers human tissues from various body regions, and has&mdash;to the best of our knowledge&mdash;one of the most significant patient-induced diversity among published works. On average, our dataset has 1.5 slide per patient, and at most 3 slides/patient. As a reference, the dataset in [Nechaev et al., 2024](https://arxiv.org/abs/2406.05074) has about 3.7 slides per patient; the dataset in [Vorontsov et al., 2024](https://arxiv.org/abs/2309.07778) 12.4 slides per patient; the dataset in [Xu et al., 2024](https://www.nature.com/articles/s41586-024-07441-w#MOESM1), 5.7 slides/patient.

### Model.
We train a vision transformer [[Dosovitskiy et al., 2021](https://openreview.net/pdf?id=YicbFdNTTy)] using a self-supervised learning framework base on
[[Zhou et al., 2023](https://openreview.net/pdf?id=ydopy-e6Dg), [Oquab et al., 2024](https://openreview.net/pdf?id=a68SUt6zFt)]. The model backbone is a `g/14` architecture with 4 registers [[Darcet et al., 2024](https://arxiv.org/pdf/2309.16588)].

Specifically, the model consists of 40 transformer blocks equipped with 24 heads in the attention layers; its embedding is of dimension of 1536. Alonside the the tile encoder from [[Xu et al., 2024](https://www.nature.com/articles/s41586-024-07441-w)], 
this is, to the best of our knowledge, the largest model backbone used in the context of computational pathology so far.

We performed our training on pods of 8 x A100 GPUs with 80Gb of memory.

## Feature extraction.

The `H-optimus-0` model checkpoint can be downloaded [here](https://public-bioptimus-eu-west-3.s3.eu-west-3.amazonaws.com/h-optimus-v0/checkpoint.pth).

The code below can be used to run inference; `H-optimus-0` expects images of size 224x224, extracted at 0.5 microns per pixel.
```python
import functools

import timm
import torch
from torchvision import transforms 


PATH_TO_CHECKPOINT = ""  # Path to the downloaded checkpoint.

params = {
    'patch_size': 14, 
    'embed_dim': 1536, 
    'depth': 40, 
    'num_heads': 24, 
    'init_values': 1e-05, 
    'mlp_ratio': 5.33334, 
    'mlp_layer': functools.partial(
        timm.layers.mlp.GluMlp, act_layer=torch.nn.modules.activation.SiLU, gate_last=False
    ), 
    'act_layer': torch.nn.modules.activation.SiLU, 
    'reg_tokens': 4, 
    'no_embed_class': True, 
    'img_size': 224, 
    'num_classes': 0, 
    'in_chans': 3
}

model = timm.models.VisionTransformer(**params)
model.load_state_dict(torch.load(PATH_TO_CHECKPOINT, map_location="cpu"))
model.eval()
model.to("cuda")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.707223, 0.578729, 0.703617), 
        std=(0.211883, 0.230117, 0.177517)
    ),
])

input = torch.rand(3, 224, 224)
input = transforms.ToPILImage()(input)

# We recommend using mixed precision for faster inference.
with torch.autocast(device_type="cuda", dtype=torch.float16):
    with torch.inference_mode():
        features = model(transform(input).unsqueeze(0).to("cuda"))

assert features.shape == (1, 1536)
```

## Experiments and evaluations.

We report experimental results on tile-level and slide-level tasks. A description of each evaluation dataset is provided in the [Appendix](#appendix), as well as the hyperparameter selection process.

### Tile-level evaluations.

The tile-level evaluations correspond to binary and multi-class classification tasks wherein we need to identify tissue types or tissue characteristics on the tiles. The number of classes in those tasks vary from 2 to 32.

To perform those classification tasks, we learn a linear classifier on top of the tile representations. 
In the table below, we report the mean accuracy (and the standard deviation) averaged over three trainings of the linear classifiers (where the randomness is due to the random splits of the cross validation and the batch of the stochastic gradient solver; see details in [Appendix](#hyperparameters)).


|              | H-optimus     | GigaPath (tile encoder) [[Xu et al., 2024](https://www.nature.com/articles/s41586-024-07441-w)] | Hibou-B [[Nechaev et al., 2024](https://arxiv.org/abs/2406.05074)] | Kaiko (B/8) [[Aben et al., 2024](https://arxiv.org/abs/2404.15217)] | Kaiko (L/14) [[Aben et al., 2024](https://arxiv.org/abs/2404.15217)] | UNI [[Chen et al., 2024](https://doi.org/10.1038/s41591-024-02857-3)] | Phikon [[Filiot et al., 2023](https://doi.org/10.1101/2023.07.21.23292757)] | <span style="color:grey">Phikon (_taken from original paper_)</span> | <span style="color:grey">Virchow (_taken from original paper_)</span> |
|:------------ |:------------- |:----------------------------------------------------------------------------------------------- |:------------------------------------------------------------------ |:------------------------------------------------------------------- |:-------------------------------------------------------------------- |:--------------------------------------------------------------------- |:--------------------------------------------------------------------------- |:------------ |:------------- |
| CAM17_WILDS  | **0.982 (0.001)** | <u>0.973 (0.001)</u>                                                                                   | 0.971 (0.006)                                                      | 0.935 (0.006)                                                       | 0.960 (0.011)                                                        | 0.972 (0.009)                                                         | 0.956 (0.005)                                                               | <span style="color:grey"> 0.971 (nan)</span>  | <span style="color:grey">0.970 (nan)</span>   |
| CRC_NORM     | **0.962 (0.002)** | 0.958 (0.003)                                                                                   | 0.945 (0.004)                                                      | <u>0.961 (0.005)</u>                                                       | 0.938 (0.005)                                                        | 0.953 (0.002)                                                         | 0.924 (0.025)                                                               | <span style="color:grey">0.958 (nan)</span>  | <span style="color:grey">0.973 (nan)</span>   |
| CRC_NO_NORM  | **0.956 (0.006)** | <u>0.945 (0.004)</u>                                                                                   | 0.936 (0.012)                                                      | 0.922 (0.026)                                                       | 0.797 (0.049)                                                        | 0.889 (0.034)                                                         | 0.839 (0.033)                                                               | <span style="color:grey">0.883 (nan)</span>  | <span style="color:grey">0.968 (nan)</span>   |
| MHIST        | **0.828 (0.005)** | <u>0.819 (0.005)</u>                                                                                   | 0.782 (0.004)                                                      | 0.783 (0.035)                                                       | 0.815 (0.010)                                                                  | **0.828 (0.008)**                                                         | 0.765 (0.010)                                                               | <span style="color:grey">0.795 (nan)</span>  | <span style="color:grey">0.834 (nan)</span>   |
| TCGA_UNIFORM | 0.843 (0.001) | 0.808 (0.000)                                                                                   | 0.780 (0.000)                                                      | **0.854 (0.001)**                                                       | <u>0.848 (0.001)</u>                                                        | 0.808 (0.000)                                                         | 0.830 (0.001)                                                               | <span style="color:grey">nan</span>          |<span style="color:grey">nan</span>           |


### Slide-level evaluations.

The slide-level evaluations correspond to binary classification tasks wherein we need to detect the presence of biomarkers (e.g., MSI or HER2) or the presence of metastasis.
These evaluations cover colorectal cancer (CRC), breast cancer (BC), and gastric cancer (GC).

To go from the tile representations to a single slide representation, we use the ABMIL approach developed by [[Ilse et al., 2018](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf)].

In the tables below, we report the average AUC-ROC averaged over 50 trainings of the ABMIL models (where the randomness is due to the random splits of the cross-validation and the random initialization of the models). We also report the standard deviation in parenthesis. As some tasks contain multiple test sets, we report performances with the following convention: `<Task name>: <test set name>`.


|                                                    | H-optimus            | GigaPath (tile encoder) [[Xu et al., 2024](https://www.nature.com/articles/s41586-024-07441-w)] | Hibou-B [[Nechaev et al., 2024](https://arxiv.org/abs/2406.05074)] | Kaiko (B/8) [[Aben et al., 2024](https://arxiv.org/abs/2404.15217)] | Kaiko (L/14) [[Aben et al., 2024](https://arxiv.org/abs/2404.15217)] | UNI [[Chen et al., 2024](https://doi.org/10.1038/s41591-024-02857-3)] | Phikon [[Filiot et al., 2023](https://doi.org/10.1101/2023.07.21.23292757)] |
|:-------------------------------------------------- |:-------------------- |:----------------------------------------------------------------------------------------------- |:------------------------------------------------------------------ |:------------------------------------------------------------------- |:-------------------------------------------------------------------- |:--------------------------------------------------------------------- |:--------------------------------------------------------------------------- |
| MSI prediction in CRC: FR-CRC-Bio                  | <u>0.828 (0.011)</u> | **0.873 (0.004)**                                                                               | 0.66 (0.02)                                                        | 0.767 (0.023)                                                       | 0.75 (0.024)                                                         | 0.788 (0.012)                                                         | 0.775 (0.013)                                                               |
| MSI prediction in CRC: PAIP 2020                   | 0.967 (0.007)        | <u>0.976 (0.004)</u>                                                                            | 0.942 (0.007)                                                      | 0.931 (0.008)                                                       | 0.939 (0.016)                                                        | 0.965 (0.004)                                                         | **0.978 (0.005)**                                                           |
| MSI prediction in GC: TCGA-STAD-Kather             | **0.846 (0.004)**    | <u>0.83 (0.003)</u>                                                                             | 0.745 (0.014)                                                      | 0.802 (0.016)                                                       | 0.7 (0.009)                                                          | 0.825 (0.005)                                                         | 0.787 (0.015)                                                               |
| Breast cancer detection in lymph nodes: Camelyon16 | **0.989 (0.003)**    | 0.977 (0.003)                                                                                   | 0.931 (0.016)                                                      | 0.924 (0.026)                                                       | 0.956 (0.013)                                                        | 0.968 (0.008)                                                         | <u>0.987 (0.006)</u>                                                        |
| Breast cancer detection in lymph nodes: SLN-Breast | 0.934 (0.006)        | **0.966 (0.003)**                                                                               | 0.9 (0.009)                                                        | 0.947 (0.011)                                                       | 0.935 (0.01)                                                         | <u>0.965 (0.008)</u>                                                  | 0.912 (0.009)                                                               |
| HER2 prediction in BC: Yale HER2                   | <u>0.828 (0.009)</u> | **0.835 (0.008)**                                                                               | 0.801 (0.009)                                                      | 0.765 (0.022)                                                       | 0.806 (0.016)                                                        | 0.804 (0.013)                                                         | 0.786 (0.021)                                                               |

## Appendix.
### Tile-level evaluations.
#### Tasks and datasets.

##### Identification of breast metastasis in lymph nodes (`CAM17_WILDS`).
This task consists in predicting the presence of tumor on histology patches of lymph nodes of patients diagnosed with BC. 

For this task, we use the `Camelyon17 WILDS` dataset [[Koh et al. 2020](https://arxiv.org/abs/2012.07421)], which is derived from the `Camelyon17` dataset [[Bandi et al. 2019](https://pubmed.ncbi.nlm.nih.gov/30716025/)]. This dataset is comprised of 455,954 images of resolution 96x96 pixels. The patches are resized to 224x224 before being fed to the different feature extractors. We use the official train/test splits and report the metrics on the test set.

##### Classification of tissue type in colorectal cancer (`CRC_NORM` and `CRC_NO_NORM`).
These tasks consist in classifying CRC images as one of nine tissue types : colorectal adenocarcinoma epithelium, adipose, cancer-associated stroma, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, and background.

The training set is comprised of 100,000 images (224x224 pixels) extracted from 86 slides collected at the NCT Biobank National Center for Tumor Diseases (Heidelberg, Germany) and the UMM pathology archive (University Medical Center Mannheim, Mannheim, Germany). The testing set comprises 7,180 images extracted from 50 patients of [TCGA-COAD](https://portal.gdc.cancer.gov/projects/TCGA-COAD) and [TCGA-READ](https://portal.gdc.cancer.gov/projects/TCGA-READ).

There exist two variants of the test set: one which is normalized with Macenko [[Macenko et al. 2009](https://ieeexplore.ieee.org/document/5193250)], and one which is unnormalized. We report the metrics on both test sets separately.

#### Colorectal polyp classification (`MHIST`).
This task consists in classifying CRC polyps as hyperplastic polyp or sessile serrated adenoma.

For this task, we use `MHIST` [[Wei et al. 2021](https://link.springer.com/chapter/10.1007/978-3-030-77211-6_2)], which is made up of 3,152 images (224x224 pixels) of colorectal polyps from the Department of Pathology and Laboratory Medicine at Dartmouth-Hitchcock Medical Center. We use the official train/test splits and report the metrics on the test set.

#####  Pan-cancer tissue classification (`TCGA_UNIFORM`).
This task consists in a pan-cancer tissue classification task, where each patch belongs to one the following 32 classes: adrenocortical carcinoma, bladder urothelial carcinoma, brain lower-grade glioma, breast adenocarcinoma, cervical  squamous cell carcinoma and endocervical adenocarcinoma, cholangiocarcinoma, colon adenocarcinoma, esophagus adenocarcinoma, glioblastoma multiforme, head and neck squamous cell carcinoma, kidney chromophobe, kidney renal clear cell carcinoma, kidney renal papillary cell carcinoma, liver hepatocellular carcinoma, lung adenocarcinoma, lung squamous cell carcinoma, lymphoid neoplasm diffuse large B cell lymphoma, mesothelioma, ovarian serous cystadenocarcinoma, pancreatic adenocarcinoma, pheochromocytoma  and paraganglioma, prostate adenocarcinoma, rectum adenocarcinoma, sarcoma, skin cutaneous melanoma, stomach adenocarcinoma, testicular germ cell tumor, thymoma, thyroid carcinoma, uterine carcinosarcoma, uterine corpus endometrial carcinoma, and uveal melanoma.

We use a subset of 271,107 256x256 pixels patches at 0.5 microns per pixel (MPP) from the `TCGA-Uniform` dataset [[Komura et al. 2022](https://pubmed.ncbi.nlm.nih.gov/35235802/)]. 


#### Hyperparameters.

For each dataset, we learn a linear classifier by minimizing the cross-entropy loss with Adam [[Kingma et al., 2014](https://arxiv.org/abs/1412.6980)] with a constant learning rate.
We select the following hyperparameters by 3-fold cross validation while minimizing the binary cross-entropy:
* Number of training epochs in {1, ..., 100}
* Learning rate in {0.001, 0.0001}
* Weight decay in {0.0, 0.001, 0.0001}


### Slide-level evaluations.
#### Tasks and datasets.

##### Microsatellite instability prediction in colorectal cancer.
This task consists in predicting microsatellite instability (MSI) from CRC slides. 

The train set used comprises 433 slides from [TCGA-COAD](https://portal.gdc.cancer.gov/projects/TCGA-COAD) and [TCGA-READ](https://portal.gdc.cancer.gov/projects/TCGA-READ) cohorts. The ground truth MSI labels were obtained from [[Liu et al 2018](https://www.cell.com/cancer-cell/fulltext/S1535-6108(18)30114-4)].

Several test sets were used. `PAIP 2020` which is a small cohort from the [Pathology AI Plaform](<http://www.wisepaip.org)>) was used as a first test set and consists of 47 CRC samples from 3 centers in Korea. `FR-CRC-Bio`, a proprietary dataset of OWKIN consisting of 727 CRC biopsies from multiple French hospitals, was used as a second test set. 

##### Metastasis detection in breast cancer.
This task consists in predicting the presence of tumor in lymph nodes of patients with breast cancer.

The train set of `Camelyon16` [[Bejnordi et al. 2017](https://pubmed.ncbi.nlm.nih.gov/29234806/)] competition was used for training and consists of 270 lymph nodes slides patients diagnosed with breast cancer. Two test sets were used: the test set of `Camelyon16` competition (129 slides), and `SLN-Breast` [[Campanella et al. 2019](https://www.nature.com/articles/s41591-019-0508-1)], a dataset composed of 130 slides from Memorial Sloan Kettering Cancer Center.

##### HER2 prediction in breast cancer.
This task consists in predicting HER2 (positive / negative) biomarker from breast cancer slides. 

[TCGA-BRCA](https://portal.gdc.cancer.gov/projects/TCGA-BRCA) cohort was used as training set, which is composed of 1,087 slides. HER2 labels were taken from [[Thennavan et al. 2021](https://pubmed.ncbi.nlm.nih.gov/35465400/)].

192 cases of HER2 positive and negative invasive breast carcinomas H&E slides from the Yale Pathology electronic database [[Farahmand el al., 2022](https://www.modernpathology.org/article/S0893-3952(22)00349-0/fulltext)] were used for testing. 

##### Microsatellite instability prediction in gastric cancer.
This task consists in predicting MSI from gastric cancer slides. We used the train/test split proposed in [[Kather et al. 2019](https://www.nature.com/articles/s41591-019-0462-y)], which notably exclude MSI-low patients from the train and test sets.

The train set (respectively test set) consists of 199 (respectively 103) slides from [TCGA-STAD](https://portal.gdc.cancer.gov/projects/TCGA-STAD).


#### Hyperparameters.

For each task, we train 10 ABMIL models [[Ilse et al., 2018](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf)] by minimizing the binary cross-entropy loss with Adam [[Kingma et al., 2014](https://arxiv.org/abs/1412.6980)] and a constant learning rate of 0.001.

We select the number of training epochs in {1, 5, 10, 15, ..., 50} by 5-fold cross validation while minimizing the binary cross-entropy.

For the sake of robustness, the above procedure is repeated 5 times with different PyTorch seeds. The values reported in the table are the average metrics of the 5*10 ABMIL models. The standard deviations reported in parenthesis are the average standard deviation over the 5 seeds.

For the sake of speed, a random subset of 3,000 tiles per slide is selected during training. For inference, a subset of 8,000 tiles is randomly selected.


#### GigaPath benchmark.
In this benchmark we used the tile encoder of GigaPath instead of using both its tile and slide level encoders. This enabled us to benchmark GigaPath on tile-level tasks, while comparing all feature extractors for slide-level tasks in the very same way, i.e., using ABMIL on top of the frozen features.

## Acknowledgements.

The following datasets from [TCIA](https://www.cancerimagingarchive.net/) were used in the benchmarks:

- Campanella, G., Hanna, M. G., Brogi, E., & Fuchs, T. J. (2019). Breast Metastases to Axillary Lymph Nodes [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.3xbn2jcc

- Farahmand, Saman, Fernandez, Aileen I, Ahmed, Fahad Shabbir, Rimm, David L., Chuang, Jeffrey H., Reisenbichler, Emily, & Zarringhalam, Kourosh. (2022). HER2 and trastuzumab treatment response H&E slides with tumor ROI annotations (Version 3) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/E65C-AM96

The results published here are part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga.

Regarding the `PAIP 2020` dataset: De-identified pathology images and annotations used in this research were prepared and provided by the Seoul National University Hospital by a grant of the Korea Health Technology R&D Project through the Korea Health Industry Development Institute (KHIDI), funded by the Ministry of Health & Welfare, Republic of Korea (grant number: HI18C0316).

## License.
Our code and model weights are released under the Apache License 2.0. See [LICENSE](./LICENSE.md) for additional details.


## Citation.

If you find this repository useful, please consider giving a star ⭐ and citation:
```
@software{hoptimus0,
  author = {Saillard, Charlie and Jenatton, Rodolphe and Llinares-López, Felipe and Mariet, Zelda and Cahané, David and Durand, Eric and Vert, Jean-Philippe},
  title = {H-optimus-0},
  url = {https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0},
  year = {2024},
}
```
### Contributions:
* Core contributors (alphabetical order): Jenatton, Rodolphe and Llinares-López, Felipe and Mariet, Zelda and Saillard, Charlie.
* Contributors (alphabetical order): Cahané, David and Durand, Eric and Vert, Jean-Philippe.
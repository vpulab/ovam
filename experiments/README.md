# Experiments

This folder contain scripts used to perform the paper experiments 4.1 and 4.2. And the ablation study of 4.4.

Folders structure:

* [evaluation/](./evaluation): Contains scripts to measure the mIoU reported in the paper tables and figures of sections 4.1, 4.2 and 4.3. Generated images, masks and annotations are not included in this repository due to their size. However, the scripts can be used to generate them or we can provide them upon request.
* [voc_simple/](./voc_simple). Generates the dataset VOC_sim with OVAM masks. These masks are used in the experiments of section 4.1 and 4.2.
* [coco_captions/](./coco_captions). Generates the dataset COCO_cap with OVAM masks. These masks are used in the experiments of section 4.1 and 4.2.
* [extract_sa/](./extract_sa). Contains scripts to extract the self-attention matrices from VOC-sim and COCO_cap dataset. These matrices are used in a post-processing step to generate the attention masks in OVAM.
* [attn2mask/](./attn2mask). Contains scripts to generate the attention masks to compare the work attn2mask. Used in the experiments of section 4.1 and 4.2.
* [generate_synthetic_dataset/](./generate_synthetic_dataset). Generate the synthetic dataset used in the experiments of section 4.3. This dataset, with 20K images is used to train a semantic segmentation model for VOC Challenge 2012.
* [filter_synthetic_dataset/](./filter_synthetic_dataset). Filter the synthetic dataset to generate the dataset used in the experiments of section 4.3. Uses clip to automatically filter images with more quality.
* [ablation_blocks_epochs/](./ablation_blocks_epochs). Perform the ablation study of section 4.4.

Disclaimer, the library OVAM have been cleaned and refactored to be more readable and easy to use. However, some scripts in this folder have not been refactored. We are working on it (minimal changes).

In addition, file [../docs/assets/tokens_VOC.npy](../docs/assets/tokens_VOC.npy) includes a dictionary with tokens optimized with the 20 VOC dataset classes, used in all experiments for the evaluation.
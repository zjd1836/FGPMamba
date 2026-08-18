# FGPMamba and ADEC-CD

This is the official repository for **FGPMamba**, a frequency–geometry prior-guided Mamba framework for building change detection, and **ADEC-CD**, a high-resolution building change detection dataset covering Egypt’s New Administrative Capital.

## Updates

* **August 2026:** The ADEC-CD dataset and its fixed geographically independent training, validation, and test partitions are publicly available.
* The FGPMamba implementation, training configurations, and evaluation scripts are being organized and will be released during the review process and no later than final acceptance.

## ADEC-CD Dataset

ADEC-CD is a binary building change detection dataset constructed from bitemporal SuperView-2 optical imagery acquired in 2018 and 2024 over Egypt’s New Administrative Capital.

The dataset focuses on large-scale construction monitoring in arid emerging-city environments, where changed buildings may be confused with bare land, concrete surfaces, roads, shadows, construction materials, and unfinished structures.

### Dataset characteristics

* **Study area:** Egypt’s New Administrative Capital and adjacent development areas
* **Acquisition years:** 2018 and 2024
* **Spatial resolution:** 0.5 m
* **Image size:** 256 × 256 pixels
* **Task:** Binary building change detection
* **Number of bitemporal image pairs:** 16,236
* **Training set:** 12,614 pairs
* **Validation set:** 1,381 pairs
* **Test set:** 2,241 pairs
* **Partition protocol:** Geographically independent multi-zone split

The dataset includes representative construction situations such as newly completed buildings, identifiable under-construction buildings, incremental renewal within built-up areas, and building demolition. These situations are descriptive rather than independent semantic classes; all annotations use binary changed-building labels.

The training, validation, and test subsets correspond to the fixed geographic partition used in the manuscript. Spatially coherent validation and test zones are separated from the training areas to reduce spatial leakage.

## Download

The complete cropped and partitioned ADEC-CD dataset can be downloaded from Baidu Netdisk:

* **File:** `ADEC-CD.zip`
* **Download link:** https://pan.baidu.com/s/1SvVkiGLo62jWUAOsfXTkDA?pwd=c61m
* **Extraction code:** `c61m`

Please retain the provided training, validation, and test partitions when reproducing the results reported in the manuscript.

## FGPMamba Code

The following materials are being verified and organized:

* FGPMamba implementation;
* model-specific training configurations;
* environment and dependency information;
* evaluation scripts;
* boundary-evaluation scripts; and
* large-scene sliding-window inference scripts.

These materials will be released in this repository during the review process and no later than final acceptance.

## License and Intended Use

ADEC-CD is released for non-commercial academic research and educational use. Users may use the dataset for research, comparison, and reproducibility studies provided that the dataset source and the corresponding paper are properly acknowledged.

Redistribution of modified versions, incorporation into commercial products, or commercial use requires prior permission from the dataset authors. Users are responsible for ensuring that their use of the dataset complies with applicable laws and institutional requirements.

A separate license file will be provided in this repository.

## Citation

The manuscript is currently under review. Citation information will be added once the paper record becomes available.

If you use ADEC-CD in your research, please cite the corresponding paper after its publication and acknowledge this repository.

## Contact

For questions regarding ADEC-CD, FGPMamba, or data access, please open an issue in this repository.

Thank you for your interest in our work.

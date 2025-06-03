## Scaling Up SAGA: Adapting 3D Segmentation for Large-Scale Aerial Datasets 
**Thesis Experiments**

This repository accompanies my master’s thesis and demonstrates experiments using [CityGaussian_v2](https://github.com/Linketic/CityGaussian ) and [SAGA (Segment Any Gaussian)](https://github.com/Jumpat/SegAnyGAussians).
These visualizations show segmentation of houses and buildings from large-scale 3D city models using the methods developed in this thesis.

<p align="center">
  <img src="https://github.com/user-attachments/assets/484ec9d4-a3b1-45d9-b4c3-11009c6baa1f" alt="Lund 50" width="48%" style="margin-right: 1%;">
  <img src="https://github.com/user-attachments/assets/e3ef0d58-bbae-481b-b227-1a6ea6e3f8fb" alt="Lund 1196" width="48%">
</p>

The image on the left shows the reconstructed 3D city model of Lund, while the right image highlights a single house automatically segmented from the city using the methods in this thesis.

**Implementation Note**

All key features and techniques described in the Methodology section of my thesis—such as lazy loading for memory efficiency, multi-GPU distributed training, chunk-based feature extraction, and multiple segmentation options (including HDBSCAN, KNN graph, and KMeans clustering)—are fully implemented in this codebase.
This repository contains the actual code used for large-scale experiments and results reported in the thesis.

**Note**: CityGaussian_v2 was used as-is without any modifications. As such, its code is not included in this repository. Please refer to the original repository for implementation details.

## Results

The full set of experiment results, including output files and visualizations for the **Lund 50** and **Lund 1196** datasets, is available at:

[View Results on Google Drive](https://drive.google.com/drive/folders/1sjZK4Xjwd6-ITdUM-L1--fAi9wwsNfLI?usp=drive_link)

All 3D results can be visualized using the [PlayCanvas Model Viewer](https://playcanvas.com/model-viewer).

## Additional Results

Experiments were also conducted using the Segment Anything in 3D Gaussians [SAGS](https://github.com/XuHu0529/SAGS) framework on the Lund 50 dataset to compare edge-aware segmentation performance. The following images demonstrate that SAGD provides improved boundary clarity, although some structures remain incomplete due to sparse point density.
<p align="center">
  <img src="https://github.com/user-attachments/assets/5e51ca3c-571c-417b-aa06-6e6e2b3e1e81" alt="SAGD Front View" width="48%" style="margin-right: 1%;">
  <img src="https://github.com/user-attachments/assets/041fad3a-0e3d-4712-b292-978b6316998b" alt="SAGD Back View" width="48%">
</p> 
The SAGS framework was tested separately using its original implementation, with only output visualizations shared in the results folder. The modified Jupyter Notebook for SAGS experiments is included in this repository.

## Acknowledgements
[CityGaussian_v2](https://github.com/Linketic/CityGaussian )
[SAGA (Segment Any Gaussian)](https://github.com/Jumpat/SegAnyGAussians)
[SAGS](https://github.com/XuHu0529/SAGS)
AI tools such as ChatGPT and Grammarly were used for language and code refinement.

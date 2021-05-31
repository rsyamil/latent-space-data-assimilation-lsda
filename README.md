# latent-space-data-assimilation-lsda

This repository supplements [**H108-0002 - Efficient Data Assimilation with Latent-Space Representations for Subsurface Flow Systems**](https://agu.confex.com/agu/fm20/meetingapp.cgi/Paper/764050) and [**Mohd-Razak *et al* (SPE Reservoir Simulation Conference, 2021)**](https://scholar.google.com/citations?user=mQUFzL8AAAAJ&hl=en). 

```
latent-space-data-assimilation 
│
└─── mnist
│   
└─── 2d-fluvial
```

Demos based on the MNIST dataset and a 2D fluvial field dataset (see folder structure) are archived in this repository.

## LSDA Workflow

LSDA performs simultaneous dimensionality reduction (by extracting salient spatial features from **M** and temporal features from **D**) and forward mapping (by mapping the salient features in **M** to **D**, i.e. latent spaces **z_m** and **z_d**). The architecture is composed of dual autoencoders connected with a regression model that are trained jointly. LSDA starts with an initial ensemble of prior models that are gradually updated, based on the mismatch between data simulated from each of the prior models, to the observed data. Once the iterative update steps are done, the information within the observed data has been assimilated into the ensemble of prior models, and they become calibrated posterior models that can reproduce the observed data. The forward mapping feature of LSDA replaces computationally prohibitive forward model (i.e. **G** as a physical simulator) especially when the models **M** are of high-fidelity and the size of the prior ensemble is large.

![Workflow](/readme/workflow.jpg)

Once the architecture is trained, the low-dimensional vectors **z_m** represent the high-fidelity models **M** and **z_d** represent the simulated data **D**. The (potentially) computationally expensive forward model **G** is now represented by the regression model that maps **z_m** to **z_d**, as an efficient proxy model. Given an observation vector **d_obs**, the ensemble of priors **z_m** is iteratively assimilated using Ensemble Smoother Multiple Data Assimilation (ESMDA). 

In practical applications, **d_obs** can be noisy and LSDA helps us to quickly obtain the ensemble of posteriors that can be accepted within the noise level, as well as understand the variations of spatial features within the posteriors, to improve the predictive power of the calibrated/assimilated models.
# latent-space-data-assimilation-lsda

This repository supplements [**H108-0002 - Efficient Data Assimilation with Latent-Space Representations for Subsurface Flow Systems**](https://agu.confex.com/agu/fm20/meetingapp.cgi/Paper/764050) and [**Mohd-Razak *et al* (SPE Reservoir Simulation Conference, 2021)**](https://scholar.google.com/citations?user=mQUFzL8AAAAJ&hl=en). 

## Prerequisites

The dataset used in this demo repository is the digit-MNIST images, **X**, with the forward model being a linear operator **G** and the resulting simulated responses denoted as **Y**. The physical system is represented as **Y=G(X)** or **D=G(M)**. More description on the dataset is available [here (in readme)](https://github.com/rsyamil/cnn-regression) and here is a [demo](https://github.com/rsyamil/dimensionality-reduction-autoencoders) on using convolutional autoencoders for dimensionality reduction of the digit-MNIST images.

![ForwardModel](/readme/forwardmodel.png)

We are interested in learning the inverse mapping **M=G'(D)** which is not trivial if the **M** is non-Gaussian (which is the case with the digit-MNIST images) and G is nonlinear (in this demo we assume a linear operator). Such complex mapping (also known as history-matching) may result in solutions that are non-unique with features that may not be consistent. Multiple history-matching approaches exist, such as gradient-based, [direct-inversion](https://github.com/rsyamil/latent-space-inversion-lsi) and ensemble-based methods. In ensemble-based methods, we start with an initial ensemble of prior models that are gradually updated, based on the mismatch between data simulated from each of the prior models, to the observed data. Once the iterative update steps are done, the information within the observed data has been assimilated into the ensemble of prior models, and they become posterior models that can reproduce the observed data. The problem with this approach is that the forward model used as the simulator (i.e. **G**) can be computationally prohibitive especially when the models **M** are of high-fidelity and the size of the prior ensemble is large.    

## Implementation

To address these challenges, LSDA performs simultaneous dimensionality reduction (by extracting salient spatial features from **M** and temporal features from **D**) and forward mapping (by mapping the salient features in **M** to **D**, i.e. latent spaces **z_m** and **z_d**). The architecture is composed of dual autoencoders connected with a regression model as shown below and is trained jointly. See [this Jupyter Notebook](https://github.com/rsyamil/latent-space-data-assimilation-lsda/blob/main/qc-demo-var.ipynb) to train/load/QC the architecture. Use [this Jupyter Notebook](https://github.com/rsyamil/latent-space-data-assimilation-lsda/blob/main/qc-demo.ipynb) to train/load/QC the architecture without the variational losses (i.e. no Gaussian prior constraint on the latent variables in variational autoencoders). Typically when the architecture has enough capacity to represent the model and data, the latent variables naturally become Gaussian and the variational loss becomes unnecessary. Note that the constraint on the latent variables can also reduce the performance of LSDA, so evaluate your dataset and check if the constraint is really necessary!

![Arch](/readme/arch.jpg)

Once the architecture is trained, the low-dimensional vectors **z_m** represent the high-fidelity models **M** and **z_d** represent the simulated data **D**. The (potentially) computationally expensive forward model **G** is now represented by the regression model that maps **z_m** to **z_d**, as an efficient proxy model. Given an observation vector **d_obs**, the ensemble of priors **z_m** is iteratively assimilated using the following ESMDA equation, where each corresponding **z_d** is obtained from the proxy model: 

![Esmda](/readme/esmda_update_eqn.png)

The LSDA workflow is illustrated here:

![Workflow](/readme/workflow.jpg)

The pseudocode is described here:

![Pseud](/readme/pseudocode.png)

Note that when the variational loss is omitted, all the loss functions simply become squared-error terms (i.e. similar to [Latent-Space-Inversion, LSI](https://github.com/rsyamil/latent-space-inversion-lsi)). 

## Demo of LSDA Workflow

The figures in this demo can be reproduced using this [notebook](https://github.com/rsyamil/latent-space-data-assimilation-lsda/blob/main/lsda-demo.ipynb). In this demonstration, assume the following reference case of digit zero (the same reference case in [LSI](https://github.com/rsyamil/latent-space-inversion-lsi)) and no Gaussian prior constraint on the latent variables **z_m** and **z_d**. The first left image in the figure below shows the reference **m** and the second shows its reconstruction from the model autoencoder. The third plot is the **d_obs** with its reconstruction from the data autoencoder and the fourth plot shows the prediction/forecast from the proxy model.   

![Ref](/readme/test_sigs_ref_regs_demo.png)

The histograms below show **z_d** (simulated and reduced from the entire initial prior ensemble) and the red line represents the latent variables **z_d_obs**. The black unfilled bars denoted as "Update" represents the **z_d** from the ensemble of posteriors, and they agree with **z_d_obs**.

![zds](/readme/test_zds_demo.png)

Similarly, the histograms below show **z_m** (reduced from the entire initial prior ensemble) and the red line represents the latent variables **z_m_ref** of the reference **m** (assume known). The black unfilled bars denoted as "Update" represents the assimilated **z_m** (i.e. posteriors).

![zms](/readme/test_zms_demo.png)

![prpost_m](/readme/test_prior_posts_demo.png)

![prpost_meanvar](/readme/prpostmeanvar.png)

![prpost_d](/readme/test_d_post_demo.png)

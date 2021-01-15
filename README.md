Code for "[Generative causal explanations of black-box classifiers](https://arxiv.org/abs/2006.13913)" by Matt O'Shaughnessy, Greg Canal, Marissa Connor, Mark Davenport, and Chris Rozell (Proc. NeurIPS 2020).

## Demo/quick start
`demo.py` contains code demonstrating the use of the generative causal explainer (GCE) class (`GCE.py`) and explanation plotting functions (`plotting.py`). The demo code reproduces Figure 3 (creating an explanation of a simple pretrained MNIST 3/8 classifier) and is the easiest place to start.

## Reproduce main paper plots

**Prerequisites**
Generating results requires only python (version 3.8.5 used), pytorch (version 1.6.0 with CUDA 10.1 used), numpy (version 1.19.1 used), scipy (version 1.5.2 used), and matplotlib (version 3.3.1 used). Using pretrained models requires CUDA, but you should be able to regenerate the results yourself without. Matlab and inkscape were used to create some final results.

**Figure 3** (global explanations of MNIST digits)
Run `make_fig3.py`. By default, this script will load a pretrained classifier from `pretrained_models/mnist_38_classifier/` and a pretrained explanatory VAE from `pretrained_models/mnist_38_gce/`.
- *To retrain the classifier:* run `train_mnist_classifier.py`, setting `dataset = 'mnist'` and `class_use = np.array([3,8])`.
- *To retrain the explanatory VAE:* set `retrain_gce = True`, optionally changing any of the parameters at the top of the file. You may also want to set `save_gce = True` and change `gce_path`.

**Figure 4** (comparison of glabal explanations with other methods)
- Left panel (local explanations from other popular techniques)
   - Run `make_fig4_left.py`. *Note:* this script requires tensorflow, keras, and scikit-image. In addition, it requires the [LIME](https://github.com/marcotcr/lime) and [SHAP](https://github.com/slundberg/shap) packages.
- Right side of plot (our local explanation)
  Run `make_fig4_right.py`. By default, the script will load a pretrained classifier from `pretrained_models/mnist_38_classifier/`, load the corresponding pretrained explanatory VAE from `pretrained_models/mist_38_gce/`, and generate plots showing global explanations.
   - *To retrain the classifier:* The default pretrained classifier was created from `train_mnist_classifier.py` with `dataset = 'mnist'` and `class_use = np.array([3,8])`. To use a different classifier, change `classifier_path` at the top of the file.
   - *To retrain the explanatory VAE:* the default pretrained VAE was created from `make_fig3.py` (be sure to uncomment the lines in this file that save the GCE object if you make changes). To use a different classifier, change `gce_path` at the top of the `make_fig4_right.py`.

**Figure 5** (quantitative results with fashion MNIST)
 - Subfigures (a-b) (information flow and reduction in classifier accuracy for fashion MNIST classes 0/3/4)
   Run `make_fig5ab.py`. By default, the script will load a pretrained classifier from `pretrained_models/fmnist_034_classifier/`, load the corresponding pretrained explanatory VAE from `pretrained_models/fmnist_034_gce/`, and compute and plot figures 5(a-b).
   - *To retrain the classifier:* The default pretrained classifier was created from `train_mnist_classifier.py` with `dataset = 'fmnist'` and `class_use = np.array([0,3,4])`. To use a different classifier, change `classifier_path` at the top of the file.
   - *To retrain the explanatory VAE:* set `retrain_gce = True`, optionally changing any of the parameters at the top of the file. You may also want to set `save_gce = True` and change `gce_path`.
 - Subfigures (c-d) (global explanation of fashion MNIST classes 0/3/4)
   Run `make_fig5cd.py`. By default, the script will load a pretrained classifier from `pretrained_models/fmnist_034_classifier/`, load the corresponding pretrained explanatory VAE from `pretrained_models/fmnist_034_gce/`, and create latent plot sweeps such as those in Figure 5(c-d) for each latent factor.
   - *To retrain the classifier:* The default pretrained classifier was created from `train_mnist_classifier.py` with `dataset = 'fmnist'` and `class_use = np.array([0,3,4])`. To use a different classifier, change `classifier_path` at the top of the file.
   - *To retrain the explanatory VAE:* set `retrain_gce = True`, optionally changing any of the parameters at the top of the file. You may also want to set `save_gce = True` and change `gce_path`.

## Reproduce appendix plots

**Figure 8** (empirical results for causal objective with linear/gaussian generative map, linear classifier)
 - Run `make_fig8.py`. This will generate some rough plots and save results to `results/fig8.mat`.
 - To create plots in paper: run first cell of `make_fig8_fig9_fig10.m`.

**Figures 9-10** (empirical results for causal/combined objective with linear/gaussian generative map, AND classifier)
 - Run `make_fig9_fig10.py`. This will generate some rough plots and save results to `results/fig9.mat`.
 - To create plots in paper: run second and third cells of `make_fig8_fig9_fig10.m`.

**Figure 11** (snapshots of parameter tuning procedure for MNIST 3/8)
Run `make_fig11.m`. This script uses pre-saved results in `results/tuning_mnist38_*.mat`. These .mat files contain additional information from the parameter turning process shown.

**Figure 12** (additional results for global explanations of MNIST digits):
Contains complete results from Figure 3; see `make_fig3.py`.

**Figure 13** (global explanations of 1/4/9 MNIST digits):
Run `make_fig13.py`. By default, this script will load a pretrained classifier from `pretrained_models/mnist_149_classifier/` and a pretrained explanatory VAE from `pretrained_models/mnist_149_gce/`.
- *To retrain the classifier:* run `train_mnist_classifier.py`, setting `dataset = 'mnist'` and `class_use = np.array([1,4,9])`.
- *To retrain the explanatory VAE:* set `retrain_gce = True`, optionally changing any of the parameters at the top of the file. You may also want to set `save_gce = True` and change `gce_path`.

**Figure 14** (zoomed global explanations of 1/4/9 MNIST digits):
Run `make_fig14.py`. By default, this script will load a pretrained classifier from `pretrained_models/mnist_149_classifier/` and a pretrained explanatory VAE from `pretrained_models/mnist_149_gce/`.
- *To retrain the classifier:* run `train_mnist_classifier.py`, setting `dataset = 'mnist'` and `class_use = np.array([1,4,9])`.
- *To retrain the explanatory VAE:* set `retrain_gce = True`, optionally changing any of the parameters at the top of the file. You may also want to set `save_gce = True` and change `gce_path`. 

**Figure 15** (detailed plots of our method as used in local comparison)
Contains complete results from Figure 4 (right); see `make_fig4.py`.

**Figure 16** (snapshots of parameter tuning procedure for 0/3/4 fashion MNIST digits)
Run `make_fig16.m`. This script uses pre-saved results in `results/tuning_fmnist034_*.mat`. These .mat files contain additional information from the parameter turning process shown.

**Figure 17** (detailed global explanations of fashion MNIST)
Contains complete results from Figure 5(c-d); see `make_fig5cd.py`.

**Figure 18** (experiments comparing VAE capacity)
- Run `make_fig18_fig19.py`, which creates the file `results/fig18.mat`.
- Run the matlab script `make_fig18.m` to create the final plot. Note that this script requires the [cbrewer](https://www.mathworks.com/matlabcentral/fileexchange/34087-cbrewer-colorbrewer-schemes-for-matlab) matlab package.

**Figure 19** (qualitative results for varying VAE capacity)
Contains qualitative results from Figure 18; see `make_fig18_fig19.py`, which creates the files `./figs/fig19/XXfilters_lambdaYYY*`.
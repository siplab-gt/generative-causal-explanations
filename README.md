Code for "[Generative causal explanations of black-box classifiers](https://arxiv.org/abs/2006.13913)" by Matt O'Shaughnessy, Greg Canal, Marissa Connor, Mark Davenport, and Chris Rozell (Proc. NeurIPS 2020). Requires only numpy (version 3.8.5 used), pytorch (version 3.8.5 used), and matplotlib (version 3.3.1 used).

## Demo/quick start
`demo.py` contains code demonstrating the use of the generative causal explainer (GCE) class (`GCE.py`) and plotting functions (`plotting.py`). The demo code reproduces Figure 3 (creating an explanation of a simple pretrained MNIST 3/8 classifier) and is the easiest place to start.

## Reproduce main paper plots
The following scripts recreate the plots in the main paper:

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

*Figure 8* (empirical results for causal objective with linear/gaussian generative map, linear classifier)
 - Run `visualize_causalobj_linear.py`. This will save results to `results/visualize_causalobj_linear.mat`.
 - To create plot: run first cell of `visualize_causalobj_plot.m`

*Figures 9-10* (empirical results for causal/combined objective with linear/gaussian generative map, AND classifier)
 - To generate data: run `visualize_causalobj_and.py`. This will save results to `results/visualize_causalobj_and.mat`.
 - To create plots: run second and third cells of `visualize_causalobj_plot.m`

*Figure 11* (snapshots of parameter tuning procedure for MNIST digits)
- (optional) To retrain classifier: Run `train_mnist_classifier.py`, setting `dataset = 'mnist'` and `class_use = np.array([3,8])`. A pretrained model is located in `pretrained_models/mnist_38_classifier/`.
- To generate data: run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plots: run `make_tuning_mnist.m`, changing the file path to the results saved by the previous step

*Figure 12* (additional results for global explanations of MNIST digits):
(optional) To retrain classifier: Run `train_mnist_classifier.py`, setting `dataset = 'mnist'` and `class_use = np.array([3,8])`. A pretrained model is located in `pretrained_models/mnist_38_classifier/`.
- To train explanatory VAE: Run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plot: run `make_mnist_qual.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models. 

*Figure 13* (global explanations of 1/4/9 MNIST digits):
- (optional) To retrain classifier: Run `train_mnist_classifier.py`, setting `dataset = 'mnist'` and `class_use = np.array([1,4,9])`. A pretrained model is located in `pretrained_models/mnist_149_classifier/`.
- To train explanatory VAE: Run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plot: run `make_mnist_qual.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models. 

*Figure 14* (zoomed global explanations of 1/4/9 MNIST digits):
- (optional) To retrain classifier: Run `train_mnist_classifier.py`, setting `dataset = 'mnist'` and `class_use = np.array([1,4,9])`. A pretrained model is located in `pretrained_models/mnist_149_classifier/`.
- To train explanatory VAE: Run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plot: run `make_mnist_qual.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models. 

*Figure 15* (detailed plots of our method as used in local comparison)
- (optional) To retrain classifier: Run `train_mnist_classifier.py`, setting `dataset = 'mnist'` and `class_use = np.array([3,8])`. A pretrained model is located in `pretrained_models/mnist_38_classifier/`.
- To train explanatory VAE: run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plot: run `comparisons_mnist_ours.py`

*Figure 16* (snapshots of parameter tuning procedure for fashion MNIST digits)
- (optional) To retrain classifier: Run `train_mnist_classifier.py`, setting `dataset = 'fmnist'` and `class_use = np.array([0,3,4])`. A pretrained model is located in `pretrained_models/fmnist_034_classifier/`.
- To generate data: run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plots: run `make_tuning_mnist.m`, changing the file path to the results saved by the previous step

*Figure 17* (detailed global explanations of fashion MNIST)
- (optional) To retrain classifier: Run `train_mnist_classifier.py`, setting `dataset = 'fmnist'` and `class_use = np.array([0,3,4])`. A pretrained model is located in `pretrained_models/fmnist_034_classifier/`.
- To train explanatory VAE: Run `CVAE.py`, set `data_type` to `fmnist`, `decoder_net` to `VAE_fMNIST`, and `classifier` to `cnn_fmnist`
- To make plots: run `make_fmnist_qual.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models.
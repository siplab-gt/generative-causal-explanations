Code for "[Generative causal explanations of black-box classifiers](https://arxiv.org/abs/2006.13913)" by Matt O'Shaughnessy, Greg Canal, Marissa Connor, Mark Davenport, and Chris Rozell (Proc. NeurIPS 2020). Requires only numpy (version 3.8.5 used), pytorch (version 3.8.5 used), and matplotlib (version 3.3.1 used).

## Demo/quick start
`demo.py` contains code demonstrating the use of the generative causal explainer (GCE) class (`GCE.py`) and plotting functions (`plotting.py`). The demo code reproduces Figure 3 (creating an explanation of a simple pretrained MNIST 3/8 classifier) and is the easiest place to start.

## Guide to make plots
The following code can be used to recreate the plots in the paper.

*Figure 3* (global explanations of MNIST digits):
- (optional) To retrain classifier: Run `train_mnist_classifier.py`, setting `dataset = 'mnist'` and `class_use = np.array([3,8])`. A pretrained model is located in `pretrained_models/mnist_38_classifier/`.
- (optional) To retrain explanatory VAE: Run `CVAE.py`, setting `data_type` to `mnist`, `gen_model` to `VAE_CNN`, `classifier` to `CNN`, `class_use` to `np.array([3,8])`, and `classifier_path` to the appropriate `.pt` file at the bottom of the script. A pretrained model for Figures 3/4 is located in `pretrained_models/mnist_38_vae_zdim8_alphadim1_lambda0.05/`.
- To make plot: run `make_mnist_qual.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models.

*Figure 4* (comparison with other methods):
- Left side of plot (local explanations from other popular techniques)
   - Run `comparisons_mnist.py`
- Right side of plot (our local explanation)
   - (optional) To retrain classifier: Run `train_mnist_classifier.py`, setting `dataset = 'mnist'` and `class_use = np.array([3,8])`. A pretrained model is located in `pretrained_models/mnist_38_classifier/`.
   - (optional) To retrain explanatory VAE: Run `CVAE.py`, setting `data_type` to `mnist`, `gen_model` to `VAE_CNN`, `classifier` to `CNN`, `class_use` to `np.array([3,8])`, and `classifier_path` to the appropriate `.pt` file at the bottom of the script. A pretrained model for Figures 3/4 is located in `pretrained_models/mnist_38_vae_zdim8_alphadim1_lambda0.05/`.
   - To make plot: run `comparisons_mnist_ours.py`

*Figure 5* (quantitative results with fashion MNIST)
 - Subfigures (a) and (b) (information flow and reduction in classifier accuracy)
   - (optional) To retrain classifier: Run `train_mnist_classifier.py`, setting `dataset = 'fmnist'` and `class_use = np.array([0,3,4])`. A pretrained model is located in `pretrained_models/fmnist_034_classifier/`.
   - [OLD] To train explanatory VAE: Run `CVAE.py`, set `data_type` to `fmnist`, `decoder_net` to `VAE_fMNIST`, and `classifier` to `cnn_fmnist`
   - (optional) To retrain explanatory VAE: Run `CVAE.py`, setting `data_type` to `fmnist`, `gen_model` to `VAE_CNN`, `classifier` to `CNN`, `class_use` to `np.array([3,8])`, and `classifier_path` to the appropriate `.pt` file at the bottom of the script. A pretrained model for Figure 5 is located in `pretrained_models/fmnist_034_vae_zdim6_alphadim2_lambda0.05/`.
   - To make plots: run `make_fmnist_quant.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models.
 - Subfigure (c) (global explanation of fashion MNIST images)
   - (optional) To retrain classifier: Run `train_mnist_classifier.py`, setting `dataset = 'fmnist'` and `class_use = np.array([0,3,4])`. A pretrained model is located in `pretrained_models/fmnist_034_vae_zdim6_alphadim2_lambda0.05/`.
   - (optional) To retrain explanatory VAE: Run `CVAE.py`, setting `data_type` to `fmnist`, `gen_model` to `VAE_CNN`, `classifier` to `CNN`, `class_use` to `np.array([3,8])`, and `classifier_path` to the appropriate `.pt` file at the bottom of the script. A pretrained model for Figure 5 is located in `pretrained_models/fmnist_034_vae_zdim6_alphadim2_lambda0.05/`.
   - To make plots: run `make_fmnist_qual.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models.
   
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
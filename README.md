#  [Matching Feature Sets for Few-shot Image Classification](https://lvsn.github.io/SetFeat/) 
 
This repository is goining to contain the pytorch implementation of Matching Feature Sets for Few-shot Image Classification [paper](https://cvpr2022.thecvf.com/) 
[presentation](https://lvsn.github.io/MixtFSL/assets/SetFeat_Poster.pdf). This paper introduces a set-based representation intrinsically builds a richer representation of images from the base classes, which can subsequently better transfer to the few-shot classes. To do so, we propose to adapt existing feature extractors to instead produce \emph{sets} of feature vectors from images. Our approach, dubbed SetFeat, embeds shallow self-attention mechanisms inside existing encoder architectures. The attention modules are lightweight, and as such our method results in encoders that have approximately the same number of parameters as their original versions. During training and inference, a set-to-set matching metric is used to perform image classification.
 
 
 
## Dependencies
In the evaluations, we used Cuda 11.0 with the following list of dependencies:
• Python 3.8.10; • Numpy 1.21.2; • PyTorch 1.9.1+cu111; • Torchvision 0.10.1; • PIL 7.0.0; • Einops 0.3.0.



## Datasets
- Download the CUB dataset from [the project webpage](http://www.vision.caltech.edu/datasets/cub_200_2011/)
- Copy the dataset to "./benchmarks/cub/"
- In the "./benchmarks/cub/" directory, run cub <code>traintestval.py</code>: python cub traintestval.py
- Feel free to copy your dataset in the benchmarks directory change and specify the directory from <code>args.py</code>

## Train 
 - Go to the main **directory** and run <code>main.py</code>: python main.py


## The project webpage
Please visit [the project webpage](https://lvsn.github.io/SetFeat/) for more information.


## 
The code will be run with **SetFeat12** by default. Feel free to change it to **SetFeat4-64** in <code>-backbone</code> to SetFeat4 in <code>args.py</code>. 

## Citation
</code><pre>
@article{afrasiyabimatching,
  title={Matching Feature Sets for Few-Shot Image Classification Supplementary Materials},
  author={Afrasiyabi, Arman and Larochelle, Hugo and Lalonde, Jean-Fran{\c{c}}ois and Gagn{\'e}, Christian}
}
</code></pre>

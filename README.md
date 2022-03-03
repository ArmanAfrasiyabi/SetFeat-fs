#  [Matching Feature Sets for Few-shot Image Classification](https://lvsn.github.io/SetFeat/) 
 
This repository is goining to contain the pytorch implementation of Matching Feature Sets for Few-shot Image Classification [paper](https://cvpr2022.thecvf.com/) 
[presentation](https://lvsn.github.io/MixtFSL/assets/SetFeat_Poster.pdf). This paper introduces a set-based representation intrinsically builds a richer representation of images from the base classes, which can subsequently better transfer to the few-shot classes. To do so, we propose to adapt existing feature extractors to instead produce \emph{sets} of feature vectors from images. Our approach, dubbed SetFeat, embeds shallow self-attention mechanisms inside existing encoder architectures. The attention modules are lightweight, and as such our method results in encoders that have approximately the same number of parameters as their original versions. During training and inference, a set-to-set matching metric is used to perform image classification.
 
 
 
## Dependencies
1. Numpy
2. Pytorch 1.0.1+ 
3. Torchvision 0.2.1+
4. PIL


## Train 
 under construction


## Datasets
- Download ... (under construction)




## The project webpage
Please visit [the project webpage](https://lvsn.github.io/SetFeat/) for more information.

## Citation
</code><pre>
@InProceedings{Afrasiyabi_2022_CVPR,
    author    = {Afrasiyabi, Arman and Larochelle, Hugo and Lalonde, Jean-Fran{\c{c}}ois and Gagn{\'e}, Christian},
    title     = {Matching Feature Sets for Few-shot Image Classification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022} 
} 
</code></pre>

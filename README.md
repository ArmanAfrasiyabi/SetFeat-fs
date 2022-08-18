#  [Matching Feature Sets for Few-shot Image Classification](https://lvsn.github.io/SetFeat/) 
 
This repository is goining to contain the pytorch implementation of Matching Feature Sets for Few-shot Image Classification which has been accepted at Conference on Computer Vision and Pattern Recognition (**CVPR**) 2022, [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Afrasiyabi_Matching_Feature_Sets_for_Few-Shot_Image_Classification_CVPR_2022_paper.pdf),
[poster](https://lvsn.github.io/SetFeat/assets/SetFeat_Poster.pdf). This paper introduces a set-based representation intrinsically builds a richer representation of images from the base classes, which can subsequently better transfer to the few-shot classes. To do so, we propose to adapt existing feature extractors to instead produce \emph{sets} of feature vectors from images. Our approach, dubbed SetFeat, embeds shallow self-attention mechanisms inside existing encoder architectures. The attention modules are lightweight, and as such our method results in encoders that have approximately the same number of parameters as their original versions. During training and inference, a set-to-set matching metric is used to perform image classification.
 

 
## Dependencies
In the evaluations, we used Cuda 11.0 with the following list of dependencies:
1. Python 3.8.10; 
2. Numpy 1.21.2; 
3. PyTorch 1.9.1+cu111; 
4. Torchvision 0.10.1; 
5. PIL 7.0.0; 
6. Einops 0.3.0.

## Note 
there is a tiny typo in page 5 in the paper "100/50/50" at the last line of the left column (inside parenthesis) should be "64/16/20". For detialed description please see [supp. mat.](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Afrasiyabi_Matching_Feature_Sets_CVPR_2022_supplemental.pdf).

## Datasets
- For dataset and backbone specifications please see table 1 and table 2 of our [supp. mat.](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Afrasiyabi_Matching_Feature_Sets_CVPR_2022_supplemental.pdf)
- Download the CUB dataset from [www.vision.caltech.edu](http://www.vision.caltech.edu/datasets/cub_200_2011/)
- Copy the dataset to "./benchmarks/cub/"
- In the "./benchmarks/cub/" directory, run cub <code>traintestval.py</code>: python cub traintestval.py
- Feel free to copy your dataset in the benchmarks directory change and specify the directory from <code>args.py</code>

## Train & Test 
 - Go to the main **directory** and run <code>main.py</code>: python main.py


## The project webpage
Please visit [the project webpage](https://lvsn.github.io/SetFeat/) for more information.


## 
The code will be run with SetFeat12* by default. Feel free to change it to **SetFeat4-64** in <code>-backbone</code> to SetFeat4 in <code>args.py</code>. 

 

## Citation
</code><pre>
@inproceedings{afrasiyabi2022matching,
  title={Matching Feature Sets for Few-Shot Image Classification},
  author={Afrasiyabi, Arman and Larochelle, Hugo and Lalonde, Jean-Fran{\c{c}}ois and Gagn{\'e}, Christian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9014--9024},
  year={2022}
}
</code></pre>

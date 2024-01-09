# LaKDNet: Revisiting Image Deblurring with an Efficient ConvNet 

  [Lingyan Ruan](http://lyruan.com), [Mojtaba Bemana](https://people.mpi-inf.mpg.de/~mbemana/), [Hans-peter Seidel](https://people.mpi-inf.mpg.de/~hpseidel/), [Karol Myszkowski](https://people.mpi-inf.mpg.de/~karol/), [Bin Chen](https://binchen.me/) 


Max-Planck-Institut fur Informatik  

> **Abstract:** *The recent advancements in Transformers for computer vision tasks have had a notable impact on the field of image restoration. This has led to the development of generic structures, such as Uformer, and Restormer, which have shown superior performance over dedicated task-specific CNNs. The success of these structures can be attributed to their ability to handle long-range interactions, which is believed to be lacking in CNNs. The aim of this paper is to address the limitations of CNN-based structures and enable them to perform image restoration tasks such as motion and defocus deblurring with comparable effectiveness to Transformers. To investigate the factors contributing to restoration performance differences, we analyze the effective receptive field (ERF) of 10 existing methods and propose a metric called ERFMeter to compare ERF across different architectures. Our analysis reveals that the global and local properties of ERF are crucial for achieving superior performance. Based on these findings, we propose a CNN structure called LaKDNet, incorporating a large kernel convolution and mixer shortcuts scheme to enhance the global and local properties of ERF in CNNs. This approach demonstrates higher efficiency than generic Transformer works, as well as existing CNNs with multi-scale-stage strategies. Specifically, we achieve a PSNR improvement of +0.80 dB / +0.67 dB over the state-of-the-art Restormer / Uformer on the GOPRO dataset. Our findings suggest that there are still rooms to improve the performance of CNN when refining the network structure towards an optimal ERF.* 
<hr />

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2302.02234) [Come soon!]




## Effective Receptive Field Analysis 
**Motion & Defocus**

<img src = "./assets/ERF_demo.png" width='60%' height ='60%'> 

## Visual Performance

<img src = "./assets/visual_performance.png" width='80%' height ='80%'> 

## Performance vs. Computational Cost
<img src = "./assets/assets_params.png" width='80%' height ='80%'> 


## Will update soon!
## Quick Demo

## Instruction on the Training and Evaluation

## Visual Result and Pre-trained Models

## Quantitive Result

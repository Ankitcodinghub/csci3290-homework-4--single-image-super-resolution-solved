# csci3290-homework-4--single-image-super-resolution-solved
**TO GET THIS SOLUTION VISIT:** [CSCI3290 Homework 4 -Single Image Super Resolution Solved](https://www.ankitcodinghub.com/product/csci3290-homework-4-single-image-super-resolution-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;92238&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSCI3290 Homework 4 -Single Image Super Resolution Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
Single Image Super Resolution

1 Introduction

Single image super resolution (SISR) is a classical image restoration problem which aims to recover a high-resolution (HR) image from the corresponding low-resolution (LR) image.

In this assignment, you’ll need to implement a super resolution convolutional neural network (SRCNN) with PyTorch. We use “Learning a Deep Convolutional Network for Image Super-Resolution” [1] as the basic reference. The basic network architecture and implementation details will be provided in the following sections. In the end, you should submit the source code and the well-trained model after your finish this assignment.

2 Implementation details 2.1 SRCNN

SRCNN uses pairs of LR and HR images to learn the mapping between them. For this purpose, image databases containing LR and HR pairs are created and used as a training set. The learned mapping can be used to predict HR details in a new image.

The SRCNN consists of the following operations:

<ol>
<li>Preprocessing: Upscales LR image to desired HR size (using bicubic interpolation).</li>
<li>Feature extraction: Extracts a set of feature maps from the upscaled LR image.</li>
<li>Non-linear mapping: Maps the feature maps representing LR to HR patches.</li>
<li>Reconstruction: Produces the HR image from HR patches.</li>
</ol>
Operations 2–4 above can be cast as a convolutional layer in a CNN that accepts the upscaled images as input, and outputs the HR image. This CNN consists of three convolutional layers:

<ul>
<li>Conv. Layer 1: Patch extraction

o 64filtersofsize3x9x9(padding=4,stride=1) o Activationfunction:ReLU

o Output: 64 feature maps</li>
<li>Conv. Layer 2: Non-linear mapping

o 32 filters of size 64 x 1x 1 (padding=0, stride=1) o Activationfunction:ReLU

o Output: 32 feature maps</li>
<li>Conv. Layer 3: Reconstruction

o 3 filter of size 32 x 5 x 5 (padding=2, stride=1) o Activationfunction:Identity

o Output:HRimage

The overall structure of SRCNN is shown in Figure 1.
</li>
</ul>
</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
Figure 1. Network Architecture of SRCNN with upscaling factor=3

In this assignment, you will need to implement a SRCNN with upscaling factor 3 in PyTorch. Let 𝑌𝑌 (𝑥𝑥)

denote this SRCNN model in the following sections.

2.2 Model Training

A typical training framework for a neural network is as follows:

<ul>
<li>Define the neural network that has some learnable parameters (or weights)</li>
<li>Iterate over a dataset of inputs</li>
<li>Process input through the network</li>
<li>Compute the loss between output and the ground truth (how far is the output from being correct)</li>
<li>Propagate gradients back into the network’s parameters</li>
<li>Update the weights of the network, typically using a simple update rule: weight = weight – learning_rate * gradient
The SRCNN is a simple feed-forward neural network. It upscaled the input LR, feeds the upscaled image through several layers one after the other, and then finally gives the output. The overall training procedure of this network is the same as the above framework. To be specific, with PyTorch, the pseudocode of training procedure for SRCNN can be described as follows:
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
𝜃𝜃

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
procedure TrainOneEpoch(𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚 𝑌𝑌 , 𝑚𝑚𝑜𝑜𝑜𝑜𝑜𝑜𝑚𝑚𝑜𝑜𝑜𝑜𝑚𝑚𝑜𝑜, 𝑜𝑜𝑜𝑜𝑡𝑡𝑜𝑜𝑛𝑛𝑡𝑡𝑚𝑚𝑜𝑜)

</div>
</div>
<div class="layoutArea">
<div class="column">
for each (𝐿𝐿𝑅𝑅 , 𝐻𝐻𝑅𝑅 ) pair in 𝑜𝑜𝑜𝑜𝑡𝑡𝑜𝑜𝑛𝑛𝑡𝑡𝑚𝑚𝑜𝑜 do 𝑖𝑖𝑖𝑖

</div>
</div>
<div class="layoutArea">
<div class="column">
𝜃𝜃

</div>
</div>
<div class="layoutArea">
<div class="column">
zero the gradient buffers of 𝑚𝑚𝑜𝑜𝑜𝑜𝑜𝑜𝑚𝑚𝑜𝑜𝑜𝑜𝑚𝑚𝑜𝑜

compute 𝑚𝑚𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑖𝑖 = 𝑌𝑌𝜃𝜃(𝐿𝐿𝑅𝑅𝑖𝑖)

compute the loss l = 𝒍𝒍𝒍𝒍𝒍𝒍𝒍𝒍_𝒇𝒇𝒇𝒇𝒇𝒇𝒇𝒇𝒇𝒇𝒇𝒇𝒍𝒍𝒇𝒇(𝑚𝑚𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑜𝑖𝑖 , 𝐻𝐻𝑅𝑅𝑖𝑖 )

back-propagate the gradients from l to the parameters 𝜃𝜃 of model 𝑌𝑌𝜃𝜃 use 𝑚𝑚𝑜𝑜𝑜𝑜𝑜𝑜𝑚𝑚𝑜𝑜𝑜𝑜𝑚𝑚𝑜𝑜 to update the parameters 𝜃𝜃

</div>
</div>
<div class="layoutArea">
<div class="column">
record the loss for training statistics [optional]

</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
Note that the actual code might differ from the pseudocode. Please check tutorial notes and PyTorch document for related APIs. Besides, we use m𝑛𝑛1ean squared error (MSE) as the 𝒍𝒍𝒍𝒍𝒍𝒍𝒍𝒍_𝒇𝒇𝒇𝒇𝒇𝒇𝒇𝒇𝒇𝒇𝒇𝒇𝒍𝒍𝒇𝒇:

</div>
</div>
<div class="layoutArea">
<div class="column">
𝑛𝑛

</div>
</div>
<div class="layoutArea">
<div class="column">
𝐿𝐿(𝜃𝜃) = �‖𝑌𝑌𝜃𝜃(𝐿𝐿𝑅𝑅𝑖𝑖) − 𝐻𝐻𝑅𝑅𝑖𝑖‖2 𝑖𝑖=1

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
where 𝑛𝑛 is the number of training samples. This loss functions can be found in PyTorch APIs. Using MSE as the loss function favors a high peak signal-to-noise ratio (PSNR). The PSNR is a widely used metric for quantitatively evaluating image restoration quality and is at least partially related to the perceptual quality. We will also use PSNR (the higher the better) to measure the performance of the trained model. The PSNR related snippets are provided in the skeleton code.

In this assignment, we use 91-Image dataset as our training dataset and Set-5 dataset as the testing dataset. The data related part is provided in the skeleton code.

Other hyperparameters related to training are listed below:

<ul>
<li>Training epoch=100; one epoch means completing one loop over whole dataset</li>
<li>Optimizer: Adam</li>
<li>Learning rate=0.0001</li>
<li>Training batch size=128; the number of inputs being feed into the network at once
Note that the above hyperparameters might not lead to reasonable performance. You are encouraged to find other possible hyperparameters to achieve better performance.

2.3 Skeleton code usage 2.3.1 Project structure

The skeleton code consists of 6 files:
</li>
</ul>
<ul>
<li>train.py: a CLI program, which contains the procedure of model training [to be completed]</li>
<li>model.py: SRCNN model [need to be completed]</li>
<li>data.py: dataset related codes</li>
<li>utils.py: helper functions</li>
<li>super_resolve.py: a CLI program, which can super resolve images given a well-trained model</li>
<li>info.py: submission info [need to be completed]
In this assignment, you are required to implement a SRCNN in PyTorch 1.2+. In order to make the skeleton code functional, you need to complete these three files in the skeleton code: train.py, model.py, info.py.

2.3.2 train.py

The usage of train.py can be describe as follows:
</li>
</ul>
</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
# train the SRCNN model using GPU, set learning rate=0.0005, batch size=256,

# make the program train 100 epoches and save a checkpoint every 10 epoches python train.py train –cuda –lr=0.0005 –batch-size=256 –num-epoch=100 –save- freq=10

# train the SRCNN model using CPU, set learning rate=0.001, batch size=128,

# make the program train 20 epoches and save a checkpoint every 2 epoches

python train.py train –lr=0.001 –batch-size=128 –num-epoch=20 –save-freq=2

# resume training with GPU from “checkpoint.x” with saved hyperparameters

python train.py resume checkpoint.x –cuda

# resume training from “checkpoint.x” and override some of saved hyperparameters

python train.py resume checkpoint.x –batch-size=16 –num-epoch=200

# inspect “checkpoint.x”

python train.py inspect checkpoint.x

</div>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
Note that the checkpoint consists of the parameters of a trained model, the state of an optimizer, and the arguments (or hyperparameters) used in current training procedure. Thus, you can use checkpoint to resume training.

2.3.3 super_resolve.py

The usage of super_resolve.py can be describe as follows:

You may use this program to perform qualitative comparison using the images inside the image_examples.zip file. This file contains LR images, upscaled images with bicubic interpolation, and ground truth (GT) HR images.

3 Grading Scheme

The assignment will be graded by the following marking scheme:

</div>
</div>
<div class="layoutArea">
<div class="column">
• •

4

</div>
<div class="column">
Code [40 points]

o Thenetworkimplementation: 20points o Thecodesfortraining: 20points

Model Training [60 points]

o Thetestingresultofyourwell-trainedmodel.Thehigherevaluationmetricsare,thehigher

your score will be.

Submission guidelines

</div>
</div>
<div class="layoutArea">
<div class="column">
# use the model stored in “checkpoint.x” to super resolve “lr.bmp”

python super_resolve.py –checkpoint checkpoint.x lr.bmp

</div>
</div>
<div class="layoutArea">
<div class="column">
You will need to submit train.py, info.py, model.py and checkpoint.pth to the Blackboard. The saved checkpoint can have various filenames, you should select one and rename it to checkpoint.pth.

You need to archive all the mentioned files in .zip or .7z format, name this archive file with your name and student ID (e.g. 1155xxxxxx_lastname_firstname.zip), and then submit this file to the Blackboard.

In case of multiple submissions, only the latest one will be considered.)

</div>
</div>
</div>

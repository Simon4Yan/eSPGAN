# VisDA2020: 4th Visual Domain Adaptation Challenge
We have released training and validation sets in https://github.com/Simon4Yan/VisDA2020. Welcome to VisDA2020!
# eSPGAN
We have updated the code of eSPGAN. To use it, you need to 1) train a source model, and 2) run espgan.py to learn an adapted model. For the first step, I use the modified codes (train_IDE_plus.py) in [Here](https://github.com/Simon4Yan/Person_reID_baseline_pytorch). You could learn your own model, and be sure to change the corresponding parts of our codes ('ft_net' in models/models.py).

Here, we also provide the PyTorch version of SPGAN. Please try this code. This code is based on https://github.com/LynnHo/CycleGAN-Tensorflow-2, thanks to their project.
You could write your own data loader to use your datasets. Of course, I notice the provided data loader is not perfect, you could use yours. 

Recently, my friend [Xiao](http://xiaoxiaosun.com/) conducted an experiment on the Synthetic data [PersonX](https://github.com/sxzrt/Instructions-of-the-PersonX-dataset), and she found that SPGAN is helpful. Specifically, Synthetic-->Market result is almost 52% in rank-1 accuracy! Moreover, eSPGAN can achieve 56% in rank-1 accuracy.

Now, we use SPGAN and eSPGAN as baselines for the [4th visda challenge](http://ai.bu.edu/visda-2020). We will release all the codes and datasets when the challenge begins. We will provide clean and easy dataset loaders to read our datasets. Both SPGAN and eSPGAN will also be included to support our challenge (and in a more convenient way). 

Thanks for your attention!

Weijian

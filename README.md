##Neural Art

This is a tensorflow implementation of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

Usage
-
Basic usage:
```
neural_style.py --style_image <image.jpg> --content_image <image.jpg>
```
Custom Weights definition:
```
neural_style.py -c ./images/brad_pitt.jpg -s ./images/picasso_selfport1907.jpg -cw 10 -sw 100 -n pitt
```
<img src=https://github.com/ioanachelu/neural-art/blob/master/images/picasso_selfport1907.jpg width=256><img src=https://github.com/ioanachelu/neural-art/blob/master/images/brad_pitt.jpg width=256><img src=https://github.com/ioanachelu/neural-art/blob/master/output/'pitt'_step_999.png width=256><br/>
<img src=https://github.com/ioanachelu/neural-art/blob/master/images/style2.jpg width=256><img src=https://github.com/ioanachelu/neural-art/blob/master/images/content1.jpg width=256><img src=https://github.com/ioanachelu/neural-art/blob/master/output/topgun_step_999.png width=256><br/>
<img src=https://github.com/ioanachelu/neural-art/blob/master/images/style1.jpg width=256><img src=https://github.com/ioanachelu/neural-art/blob/master/images/tiger.jpg width=256><img src=https://github.com/ioanachelu/neural-art/blob/master/output/neural_art_step999.png width=256><br/>

Options
-
* ```--content [-c] ```: Content image path. Default is './images/content.jpg'
* ```--style [-s] ```:  Style image path. Default is './images/style.jpg'
* ```--iters [-i] ```:  Number of steps/iterations. Default is 1000
* ```--output_dir [-o] ```: Output directory. Default is './output'
* ```--content_weight [-cw] ```: Content weight. Default is 5e0
* ```--style_weight [-sw] ```: Style weight. Default is 1e2
* ```--tv_weight [-tvw] ```: Total variation denoising weight. Default is 1e-3
* ```--output_image [-n] ```: Output image name. Default is 'neural_art'
* 

Implementation details
-
Images are initialized with white noise and optimised with Adam Optimizer.

We perform style reconstructions using the conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1 layers and content reconstructions using the conv4_2 layer. The style layers have equal weights.

The feature maps are extracted using a pretrained VGG network from [Caffe](http://caffe.berkeleyvision.org/). The weights are imported using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) after updating the models from [Model Zoo](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md) with ```upgrade_net_proto_text``` and ```upgrade_net_proto_binary```




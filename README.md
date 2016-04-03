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
Style Image: <img src=https://github.com/ioanachelu/neural-art/blob/master/images/picasso_selfport1907.jpg width=256><br/>
Content Image: <img src=https://github.com/ioanachelu/neural-art/blob/master/images/brad_pitt.jpg width=256><br/>
Generated Image: <img src=https://github.com/ioanachelu/neural-art/blob/master/output/'pitt'_step_999.png width=256><br/>


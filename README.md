##Neural Art

This is a tensorflow implementation of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

Usage
-
Basic usage:
```
neural_style.py --style_image <image.jpg> --content_image <image.jpg>
```
Picasso
![picasso]()
```
neural_style.py -c ./images/brad_pitt.jpg -s ./images/picasso_selfport1907.jpg -cw 10 -sw 100 -n 'pitt'
```


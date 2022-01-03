# README

#### Organization

* Main.py includes main function. 

* Utils.py includes some helper function.

* Match.py includes two matching algorithm.

As explained in report, there are 4 jupyter notebooks for different images.

* Part1 contains: Identity filter image, blur filter image, texture transfer with k=1, super-resolution flower image, re-color image.

* Part2 contains: Artistic filter evening skyline image with k=0.5, luminance remap off and on. Artistic filter flower and leaves image.
* Part3 contains: Texture transfer with k=0.5, failed super-resolution fluff image. Texture-by-number image.
* Part4 contains: Artistic filter evening skyline image with k=5, 20. Failed artistic shore image. Artistic filter sunset image with k=2, 0.5.

#### To run the code

Install these packages:

* “matplotlib” for image display.

* “cv2” for reading and converting images between different color spaces.

* “numpy” for computation.

* “imageio” for write image onto disks.

* “pyflann” for ANN search algorithm.

Then run the jupyter notebooks. 

Generating a image usually takes about 2 hours, so **clear outputs only when necessary**. 
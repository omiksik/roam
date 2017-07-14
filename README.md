ROAM: a Rich Object Appearance Model with Application to Rotoscoping
====

# ABOUT

This code implements the state-of-the-art video-segmentation / rotoscoping tool described in the CVPR 2017 paper **ROAM: a Rich Object Appearance Model with Application to Rotoscoping** by Ondrej Miksik, Juan-Manuel Perez-Rua, Philip H. S. Torr and Patrick Perez. 

```
@inproceedings{miksik2017roam,
  author = {Ondrej Miksik and Juan-Manuel Perez-Rua and Philip H.S. Torr and Patrick Perez},
  title = {ROAM: a Rich Object Appearance Model with Application to Rotoscoping},
  booktitle = {CVPR},
  year = {2017}
}
```
### Contacts

For any questions about the code or the paper, feel free to contact us.
More info can be found on the project page: http://www.miksik.co.uk/projects/rotoscoping/roam.html


# HOW TO COMPILE

### Prerequisites

The code is written in C++11 and can be compiled and run even on machines without any GPU (see below). 
However, we strongly recommend using `CUDA` to achieve faster run-times.

The only prerequisite is `OpenCV 3.0` or newer.

We provide a CMake project for easy compilation on any platform. 
The code was tested on Windows 10, Ubuntu 16.04, and Fedora 25 with `Visual Studio 2013` or newer and `g++ 6.3.1` or newer.

### General steps

To compile the system, use the standard cmake approach:

  1. Clone ROAM into `/roam`, e.g.:

     ```
     $ git clone git@github.com:omiksik/roam.git roam
     ```

  2. Go to the downloaded folder

     ```
     $ cd roam
     ```

  3. Create a build directory

     ```
     $ mkdir build
     ```

  4. Run your CMake tool, either 

     ```
     $ cmake ..
     ```
     or use `CMake GUI`.


  5. Compile it. On linux, use standard

     ```
     $ make -j4
     ```

     On windows, open the solution (`build/ROAM.sln`) in Visual Studio and build all in release mode.

  6. Done! Everything is built in the `build/bin` directory. See below how to run ROAM.

### Using CUDA
The default option is compilation without CUDA to ensure the code would compile without any problems on any machine.

However, we strongly recommend using GPU to achieve faster run-times (by a factor of 10 on some machines).

Prerequisite are the `CUDA` drivers and SDK. 
Then, set the `WITH_CUDA` option to `ON` in CMake

   ```$ cmake -DWITH_CUDA=ON ..```

and compile.


# HOW TO RUN

ROAM is built in `build/bin` directory.
The main application is `roam_cli`
```roam_cli --ini=${FIRST_FRAME_BINARY_MASK} --seq=${LIST_OF_FRAMES} --con=${PARAMETERS} --out=${OUTPUT_DIR} --win=${ZERO_PADDING_SIZE}```

The required inputs are:

- ${FIRST_FRAME_BINARY_MASK} - path to the segmentation mask of the first frame. ROAM will track the object delineated by this mask starting from the first file of the video sequence.

- ${LIST_OF_FRAMES} - path to the text file containing list of all video images for a particular sequence. On linux, this file can be generated e.g. by `$ ls -1v .`

- ${PARAMETERS} - path to the YAML file with the configuration of the tracker. Examples can be found in the YAML folder of the source code.

- ${OUTPUT_DIR} - path to the output directory

- ${ZERO_PADDING_SIZE} - optional


### Example

1. Download the DAVIS dataset from [here](https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip). 

2. Uncompress the file, e.g. with linux command 

  ```$ unzip DAVIS-data.zip -d ~/Documents/DAVIS```

3. Go to the directory with annotated masks and video sequences.

  ```$ cd ~/Documents/DAVIS```

4. Create a text file with the sequence you want to try, for instance: 

```
  $ cd ~/Documents/DAVIS/JPEGImages/480p/blackswan/
  $ ls -1v *.jpg > list.txt
```

5. Create an output folder and run it

```
  $ mkdir roam_output
  $ roam_cli --ini="~/Documents/DAVIS/Annotations/480p/blackswan/00000.png" --seq="~/Documents/DAVIS/JPEGImages/480p/blackswan/list.txt" --con=yaml/default.yaml --out=~/Documents/DAVIS/JPEGImages/480p/blackswan/roam_output/
```

Roam will process the sequence and produce the object masks in the directory `roam_output`.


# Other 

If you use re-parametrization, you should cite Vladimir Kolmogorov's GraphCut implementation.

```
@article{Boykov01pami,
  author = {Yuri Boykov and Vladimir Kolmogorov},
  title = {An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision},
  journal = {T-PAMI},
  year = {2001},  
}
```


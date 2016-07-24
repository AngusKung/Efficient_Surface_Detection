This is the implementation of the paper "Efficient Surface Detection for Augmented Reality on 3D Point Clouds"
by Yen-Cheng Kung, Yung-Lin Huang and Shao-Yi Chien.

To run this code,<br />
PCL 1.7(Point Cloud Library) and its supplementary is required,<br />
but PCL 1.8 is recommended.<br />

* Usage:<br />
mkdir build<br />
cd build<br />
cmake ..<br />
make<br />
./EXE [options as printed]


If you use this source code, please cite the paper: "Efficient Surface Detection for Augmented Reality on 3D Point Clouds" by Yen-Cheng Kung, Yung-Lin Huang, Shao-Yi Chien, <br />
which can be found here:
http://dl.acm.org/citation.cfm?id=2949058

A few comments can be found in the codes for better understanding.<br />
In short,<br />
ESD.cpp stands for Efficient Surface Detection, which include all the implementations of the paper.<br />
viewerESD.cpp is a viewer for XYZRGBL, since PCL does not provide a viewer in this format. Useful for viewing the result after ESD.cpp. <br />
extraPlanarRefinements.cpp allow multiple frames, ex: frames from a video, to be recombined. This file is not included in default usage and some work still need to be done.<br />
For any further question,<br />
please feel free to contact me by "angusthefrog@gmail.com".<br />
Or simply open an issue here is also welcome.<br />

# Simultaneous Localization and Mapping (SLAM) in a single camera system

This paper discusses a way of solving the Simultaneous Localization and Mapping (SLAM)
problem using only a single camera system. While this is challenging because of limited 
amounts of information available from a single camera, it provides benefits in terms of cost, 
weight, and speed of processing. The paper starts by breaking the problem down into three 
topics: target/video tracking, camera calibration/mapping and probabilistic filtering. For each 
of these topics, a suitable method is decided upon and an algorithm implemented in Python. 
The algorithms are tested on a combination of dataset videos as well as simulated scenarios and 
the quality of the algorithms discussed. Issues and challenges for each section are explored and 
discussed, and the algorithms iterated on to create a system that is effective under difficult 
scenarios. The result is a high-quality video tracking algorithm and SLAM system that works 
well using simulated data. Reasonable extensions to the work are suggested to develop upon 
the findings.

![Model convergence](https://github.com/isaactallack/Single-Camera-SLAM/blob/main/images/convergence.png?raw=true)

# Gesture Classification using Kinect

Kinect skeletal frame based gesture classfication<br>


[Click for Youtube video:
<img src="https://github.com/ElliotHYLee/GestureClassifier/blob/master/Images/Capture.JPG" width="400">](https://www.youtube.com/watch?v=NoJuUvAMqN4)


## Prereq.s

Tensorflow

python libraries:

```
chmod +x pythonReady.sh
yes "yes" | sudo sh pythonReady.sh
```

## run

```
python main.py Train
python main.py Test
```


## ToDo
- Add the "ghost" class in the data set.
- plot the result in python.
- Apply CNN based on raw RGB images.
- Apply face recognition for operator identification.

## Problem Setup

MS Kinect usually returns the skeletal coordinates as the first picture.
<ul>
<img src="https://github.com/ElliotHYLee/GestureClassifier/blob/master/Images/result_human_new.jpg" width="400">

 However, it sometimes sees "Ghosts" as shown in this second picture.

<img src="https://github.com/ElliotHYLee/GestureClassifier/blob/master/Images/ghostSkeleton.JPG" width="400">

So, it is necessary to tell if a skeletal stream is from actual human or not. Fortunately, pattern-wise, the human and "ghost" skeletons look different. So, a single hidden layer NN can easily calssify the differences.<br>

Next, the four different gestures are fed as well. Idle, move forward, move back, and takeoff/landing. As a result, I needed only 5 classes, the last class can be simply "ghost" patterns.

- Idle
<img src="https://github.com/ElliotHYLee/GestureClassifier/blob/master/Images/idel.jpg" width="400">

- Move Forward
<img src="https://github.com/ElliotHYLee/GestureClassifier/blob/master/Images/proceed_gesutre.jpg" width="400">

- Move Back
<img src="https://github.com/ElliotHYLee/GestureClassifier/blob/master/Images/retreat_g.jpg" width="400">

- Takeoff/landing
<img src="https://github.com/ElliotHYLee/GestureClassifier/blob/master/Images/takeoff.JPG" width="400">

</ul>


## Results
Accuracy= 0.90
<ul>
<img src="https://github.com/ElliotHYLee/GestureClassifier/blob/master/Images/Test.jpg" width="400">
<img src="https://github.com/ElliotHYLee/GestureClassifier/blob/master/Images/Train.jpg" width="400">
</ul>

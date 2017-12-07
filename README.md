# realtime_person_object_detector

This project uses a realtime webcam to track objects, in this case a person or people. The Display is triggered on a person
moving into the camera field of view. It will then show the person on a black background. This is performed using a CV2 masking operation so that only the subject is shown. If the person leaves the camera view it will show the direction of movement and hold the last frame on screen of the last person to leave the field of view. It will display more than 1 person at a time if they appear in the field of view. As this is real time object detection with more than 90 catagories it requires a GPU in this case I used a GTX1080Ti using the faster_rcnn_resent_coco_11_06_2017 model which is more accurate than SSD but power hungry on GPU resources.
The object bounding boxes have been removed from the captured onject but can be uncommented back in. As this uses the tensorflow
object recognition API there are many dependencies and setup is not trivial. Please refer to:

https://github.com/tensorflow/models/tree/master/research/object_detection


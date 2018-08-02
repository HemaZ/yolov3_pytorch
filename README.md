## YOLOv3 Implementation using Pytorch

Weights and Config files  are Downloaded from Darknet 

https://pjreddie.com/darknet/yolo/

To run detection on Video 

`python3 video.py --video videofile.mp4 `

the output video file will be saved as 'outpy.avi'

`python3 detect.py --images imgfile.jpg --det outputFolder`

you can also run the detection on image or folder of images

`python3 detect.py --images inputfolder --det outputFolder`

![5b634be188030](https://i.loli.net/2018/08/03/5b634be188030.png)

https://www.youtube.com/watch?v=3sJEPETVF64



| CMD Argument | Function                                      |
| ------------ | --------------------------------------------- |
| --bs         | Batch Size                                    |
| --confidence | Object Confidence to filter predictions       |
| --nms_thresh | NMS Threshhold                                |
| --cfg        | CFG File                                      |
| --weights    | Darknet Weights File                          |
| --reso       | Input resolution of the network (Default 416) |
| --video      | Video File to process                         |
| --images     | Image / Directory containing images           |

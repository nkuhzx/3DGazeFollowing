import os
import numpy as np

# Data sets Structure
# |--RGBDAttentionDataset
#    |-- RGB
#        |-- D1S1A001
#        |-- D1S1A002
#        |-- D1S1A003
#        |-- ...
#    |-- Depth
#        |-- D1S1A001
#        |-- D1S1A002
#        |-- D1S1A003
#        |-- ...
#    |-- annotation.txt

class RGBDAttention(object):

    def __init__(self,dataset_dir="RGBDAttentionDataset"):

        super(RGBDAttention, self).__init__()
        # the camera parameter under the rgb camera frame

        self.rgb_dir=os.path.join(dataset_dir,"RGB")

        self.depth_dir=os.path.join(dataset_dir,"DEPTH")

        self.annofile=os.path.join(dataset_dir,"annotation.txt")

        # camera parameters
        self.fx=456.4949875
        self.fy=413.90917324216315
        self.cx=333.5563375
        self.cy=225.4449186600704
        self.R=np.array([[1,0,0],[0,1,0],[0,0,1]])

        self.camera_para=[self.cx,self.cy,self.fx,self.fy,self.R]


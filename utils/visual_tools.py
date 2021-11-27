import numpy as np
from PIL import Image

import open3d

class VisualTool(object):

    def __init__(self,interval=20):

        super(VisualTool).__init__()

        self.interval=interval
        self.flip_transform=[[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]

        # init the pcd show
        self.pcd_show=open3d.io.read_point_cloud('modelpara/example.pcd')

        # init the ground truth gaze line
        self.linespace=self.xy_linespace()

        gt_points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        gt_points = self.constructLineset(gt_points)

        gt_lines = [[0, 1], ]
        for i in range(0, interval):
            corre_list = [i + 2, i + interval + 2]
            gt_lines.append(corre_list)

        self.gt_lines=gt_lines

        gt_colors = [[1, 0, 0] for _ in range(len(gt_lines))]
        self.gt_line_set_show = open3d.geometry.LineSet()

        self.gt_line_set_show.lines = open3d.utility.Vector2iVector(gt_lines)
        self.gt_line_set_show.points = open3d.utility.Vector3dVector(gt_points)
        self.gt_line_set_show.colors = open3d.utility.Vector3dVector(gt_colors)

        # init the pred gaze line
        pred_points = np.array([[0, 0, 0], [1, 1, 2]], dtype=np.float32)
        pred_points=self.constructLineset(pred_points)

        pred_lines = [[0, 1], ]
        for i in range(0, interval):
            corre_list = [i + 2, i + interval + 2]
            pred_lines.append(corre_list)

        self.pred_lines=pred_lines
        pred_colors = [[0, 1, 0] for _ in range(len(pred_lines))]
        self.pred_line_set_show = open3d.geometry.LineSet()

        self.pred_line_set_show.lines = open3d.utility.Vector2iVector(pred_lines)
        self.pred_line_set_show.points = open3d.utility.Vector3dVector(pred_points)
        self.pred_line_set_show.colors = open3d.utility.Vector3dVector(pred_colors)

        # flip for show
        self.flip_for_show()

        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name='PCD')
        self.vis.add_geometry(self.pcd_show)
        self.vis.add_geometry(self.gt_line_set_show)
        self.vis.add_geometry(self.pred_line_set_show)

        self.rgbimg=None
        self.points_3d=None

    def constructPcd(self,rgbimg_path,depimg_path,camera_para):

        cx,cy,fx,fy,rm=camera_para

        rgbimg=Image.open(rgbimg_path)
        rgbimg = rgbimg.convert('RGB')

        width, height = rgbimg.size

        depthimg=Image.open(depimg_path)

        rgbimg=np.array(rgbimg)
        depthmap=np.array(depthimg)/1000.0

        space_DW = np.linspace(0, width-1, width)
        space_DH = np.linspace(0, height - 1, height)
        [space_xx, space_yy] = np.meshgrid(space_DW, space_DH)

        space_X = (space_xx - cx) * depthmap / fx
        space_Y = (space_yy - cy) * depthmap / fy
        space_Z = depthmap

        points_3d=np.dstack([space_X,space_Y,space_Z])
        points_3d=points_3d.reshape([-1,3])
        points_3d=np.dot(rm,points_3d.T)
        points_3d=points_3d.T
        points_3d=points_3d.reshape([width,height,3])

        rgbimg=rgbimg.reshape([-1,3])
        rgbimg=rgbimg/255.0
        points_3d=points_3d.reshape([-1,3])
        points_3d=points_3d.astype(np.float32)

        self.rgbimg=rgbimg
        self.points_3d=points_3d


    def constructGazeline(self,eye_3d,gt_gazepoint,pred_gazepoint):

        gt_gazeline=np.vstack([eye_3d,gt_gazepoint])
        pred_gazeline=np.vstack([eye_3d,pred_gazepoint])

        self.gt_lines=self.constructLineset(gt_gazeline)
        self.pred_lines=self.constructLineset(pred_gazeline)


    def visualization(self):


        self.pcd_show.points=open3d.utility.Vector3dVector(self.points_3d)
        self.pcd_show.colors=open3d.Vector3dVector(self.rgbimg)

        self.pred_line_set_show.points=open3d.utility.Vector3dVector(self.pred_lines)
        self.gt_line_set_show.points=  open3d.utility.Vector3dVector(self.gt_lines)

        self.flip_for_show()

        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()
        # cv2.waitKey(100)


    # construct the line set
    def xy_linespace(self,radius=0.02,interval=20):

        radius = 0.02
        interval = 20
        temp_x = np.linspace(-radius, radius, interval, False)
        temp_y = np.power(radius, 2) - np.power(temp_x, 2)
        temp_y[temp_y < 0] = 0
        temp_y = np.sqrt(temp_y)

        return temp_x, temp_y

    def constructLineset(self,org_point):

        ls_x,ls_y=self.linespace

        ls_x_eye=ls_x+org_point[0][0]
        ls_x_gaze=ls_x+org_point[1][0]

        ls_y_eye=ls_y+org_point[0][1]
        ls_y_gaze=ls_y+org_point[1][1]
        add_points_eye=np.vstack([ls_x_eye,ls_y_eye])
        add_points_gaze=np.vstack([ls_x_gaze,ls_y_gaze])
        add_points=np.concatenate([add_points_eye,add_points_gaze],axis=1)


        test_z=np.zeros((1,add_points.shape[1]))
        test_z[:,0:self.interval]=org_point[0][2]
        test_z[:,self.interval:]=org_point[1][2]
        add_points=np.concatenate([add_points,test_z])
        add_points=np.swapaxes(add_points,0,1)
        points=np.concatenate([org_point,add_points])

        return points

    def flip_for_show(self):

        self.pcd_show.transform(self.flip_transform)
        self.gt_line_set_show.transform(self.flip_transform)
        self.pred_line_set_show.transform(self.flip_transform)

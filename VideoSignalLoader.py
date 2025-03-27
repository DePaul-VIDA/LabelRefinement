import os.path
from torch.utils.data import Dataset, DataLoader
import cv2
import cv2
import numpy as np
import torch
from inference import write_header, write_record
from fileloader import load_people

from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow

from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result, vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo
from mmdet.apis import init_detector, inference_detector
import mmcv

try:
    from mmtrack.apis import inference_mot
    from mmtrack.apis import inference_vid
    from mmtrack.apis import init_model as init_tracking_model

    has_mmtrack = True
except (ImportError, ModuleNotFoundError):
    has_mmtrack = False

from filedata import FileData


class VideoFileLoader:

    def load_video(self, path, vid_name):
        if torch.cuda.is_available():
            device = "cuda"
            # parallel between available GPUs
        joint_file_path = path + '/joints/' + vid_name + '.csv'
        video_path = path + '/' + vid_name

        if os.path.isfile(joint_file_path):
           self.load_exsisting_video(joint_file_path, vid_name, video_path)
        else:
           self.load_new_video(joint_file_path, vid_name, video_path)

    def load_new_video(self, joint_file_path, vid_name, video_path, device):
        fd = FileData()
        tracked_persons, fd.frame_count = load_people(joint_file_path, fd)
        config_file = 'ModelSetup/mmflow/Config/configs/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.py'
        checkpoint_file = 'ModelSetup/mmflow//checkpoints/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'
        device = "cpu"
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = 0
        end_frame = total_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_valid, frame_0 = cap.read()
        flow_images = []
        frame_1 = cv2.resize(frame_0, (640, 360), interpolation=cv2.INTER_CUBIC)
        # init a model
        model = init_model(config_file, checkpoint_file, device=device)
        end = end_frame - start_frame
        counter = 0

        while frame_valid and counter < end:
            # read the next frame
            frame_valid, frame_2 = cap.read()

            if not frame_valid:
                break
            frame_2a = cv2.resize(frame_2, (640, 360), interpolation=cv2.INTER_CUBIC)
            flow_result = inference_model(model, frame_1, frame_2a)
            flow_images.append(flow_result)
            counter += 1


    def load_new_video(self, joint_file_path, vid_name, video_path, device):
        config_file = 'ModelSetup/mmflow/Config/configs/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.py'
        checkpoint_file = 'ModelSetup/mmflow//checkpoints/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'
        flow_images = []

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = 0
        end_frame = total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_valid, frame_0 = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_1 = cv2.resize(frame_0, (640, 360), interpolation=cv2.INTER_CUBIC)

        counter = 0
        frame_index = 1
        end = end_frame - start_frame

        # init a model
        model = init_model(config_file, checkpoint_file, device=device)

        config_file = 'ModelSetup/mmtrack/Config/configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
        checkpoint_file = 'ModelSetup/mmtrack/checkpoints/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'
        model_track = init_tracking_model(config_file, checkpoint_file, device=device)

        config_file = 'ModelSetup/mmpose/Config/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
        checkpoint_file = 'ModelSetup/mmpose/checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
        model_pose = init_pose_model(config_file, checkpoint_file, device=device)

        # build the pose model from a config file and a checkpoint file
        pose_model = init_pose_model(config_file, checkpoint_file, device=device)
        dataset = pose_model.cfg.data['test']['type']
        dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)

        frame_1 = cv2.resize(frame_0, (640, 360), interpolation=cv2.INTER_CUBIC)
        fl = open(joint_file_path, 'w')
        write_header(fl, vid_name)

        counter = 0
        frame_index = 1
        end = end_frame - start_frame
        return_heatmap = False
        output_layer_names = None
        video_index = 0
        while frame_valid and counter < end:
            # read the next frame
            frame_valid, frame_2 = cap.read()

            if not frame_valid:
                break
            frame_2a = cv2.resize(frame_2, (640, 360), interpolation=cv2.INTER_CUBIC)
            flow_result = inference_model(model, frame_1, frame_2a)

            track_result = inference_mot(model_track, frame_2a, frame_id=counter)
            boxes = self.process_mmtracking_results(track_result)

            pose_results, returned_outputs = inference_top_down_pose_model(pose_model, frame_2a,
                                                                           boxes, bbox_thr=None, format='xywh',
                                                                           dataset=dataset,
                                                                           dataset_info=dataset_info,
                                                                           return_heatmap=return_heatmap,
                                                                           outputs=output_layer_names)

            if not frame_valid:
                break
            write_record(fl, frame_index, frame_index / fps, pose_results, flow_result)
            flow_images.append(flow_result)
            frame_index += 1
            frame_1 = frame_2a
            counter += 1
        video_index += 1
        cap.release()
        fl.close()
        fd = FileData()
        tracked_persons, fd.frame_count = load_people(joint_file_path, fd)

        return flow_images, fd

    def process_mmtracking_results(mmtracking_results):
        """Process mmtracking results.

        :param mmtracking_results:
        :return: a list of tracked bounding boxes
        """
        person_results = []
        # 'track_results' is changed to 'track_bboxes'
        # in https://github.com/open-mmlab/mmtracking/pull/300
        if 'track_bboxes' in mmtracking_results:
            tracking_results = mmtracking_results['track_bboxes'][0]
        elif 'track_results' in mmtracking_results:
            tracking_results = mmtracking_results['track_results'][0]

        for track in tracking_results:
            person = {}
            person['track_id'] = int(track[0])
            person['bbox'] = track[1:]
            person['bbox'][2] = person['bbox'][2] - person['bbox'][0]
            person['bbox'][3] = person['bbox'][3] - person['bbox'][1]
            person_results.append(person)

        return person_results

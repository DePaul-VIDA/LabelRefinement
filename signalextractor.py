# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import mmengine
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown

from mmdet.apis import inference_mot, init_track_model
from mmpose.evaluation.functional import nms
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline
from mmdet.apis import inference_detector, init_detector
import torch
import cv2
import numpy as np


def write_header(fl, vid_name):
    fl.write('video name: ')
    fl.write(vid_name)
    fl.write('\n')
    fl.write('frame_index, vid_time (s), person_count, ')
    for i in range(0, 25):
        fl.write('{per}, {per}_x1, {per}_y1, {per}_x2, {per}_y2, {per}_prob, '.format(per='p_' + str(i)))
        for j in range(0, 17):
            fl.write('{kp}_x, {kp}_y, {kp}_prob, '.format(kp='kp_' + str(j + 1)))
    fl.write('\n')


def write_record(fl, frame_index, frame_time_est, pose_results, all_boxes):
    TRACK_ID_INDEX = 5
    fl.write(str(frame_index))
    fl.write(', ')
    fl.write(str(round(frame_time_est, 3)))
    fl.write(', ')
    count = 0
    if pose_results != None:
        fl.write(str(len(pose_results)) + ', ')
        for j in range(0, len(pose_results)):
            pose = pose_results[j]
            bbox = all_boxes[j]
            fl.write(str(bbox[TRACK_ID_INDEX]))
            fl.write(', ')
            for i in range(0, 5):
                fl.write(str(bbox[i]))
                fl.write(', ')
            for i in range(0, 17):
                fl.write(str(pose.pred_instances.keypoints[0][i][0]))
                fl.write(', ')
                fl.write(str(pose.pred_instances.keypoints[0][i][1]))
                fl.write(', ')
                fl.write(str(pose.pred_instances.keypoint_scores[0][i]))
                fl.write(', ')
            count += 1
            if count == 25:
                break
    else:
        fl.write('0, ')
    fl.write('\n')
    fl.flush()


def generate_sigs(vid_list, vid_path, out_path):

    if torch.cuda.is_available():
        device = "cuda"
        # parallel between available GPUs
    else:
        device = "cpu"
        # change key names for CPU runtime

    with torch.no_grad():
        #load models -- mmdet and mmpose
        init_default_scope('mmdet')
        # build the model from a config file and a checkpoint file




        det_model = init_track_model(
            './ModelSetup/bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval.py',
            './ModelSetup/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth',
            None,
            None,
            device=device)

        #detector = init_detector('./ModelSetup/rtmdet_m_640-8xb32_coco-person.py', './ModelSetup/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth', device)
        det_model.cfg = adapt_mmdet_pipeline(det_model.cfg)
        pose_estimator = init_pose_estimator(
            './ModelSetup/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py',
            './ModelSetup/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
            device=device,cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

        # pose_estimator = init_pose_estimator(
       #     './ModelSetup/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py',
       #     './ModelSetup/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth',
       #      device=device,cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

        vids = len(vid_list)
        count = 1
        for vid_name in vid_list:
            full_video_path = vid_path + '/' + vid_name
            data_file_name = out_path + '/' + vid_name + '.csv'

            cap = cv2.VideoCapture(full_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(vid_name + ' - Video ' + str(count) + ' of ' + str(vids) + ' - ' + str(total_frames) + ' frames\n')
            count += 1

            if os.path.exists(data_file_name) or not os.path.exists(full_video_path):
                continue

            fl = open(data_file_name, 'w')
            write_header(fl, vid_name)


            frame_valid, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)

            frame_r = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
            frame_index = 0

            while frame_valid and frame_index < total_frames:
                if not frame_valid:
                    break

                det_result = inference_mot(det_model, frame_r, frame_id=frame_index, video_len=total_frames)
                #pose result
                all_boxes = []
                for sam in det_result.video_data_samples:
                    pred_instance = sam.pred_track_instances.cpu().numpy()
                    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
                    bboxes = np.concatenate((bboxes, pred_instance.instances_id[:, None]), axis=1)
                    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                                   pred_instance.scores > 0.05)]
                    nms_res = nms(bboxes, 0.75)
                    all_boxes.extend(bboxes[nms_res, :6])
                    bboxes_ps = bboxes[nms_res, :4]
                if len(all_boxes) > 0:
                    pose_results = inference_topdown(pose_estimator, frame, bboxes_ps)
                    write_record(fl, frame_index, frame_index/fps, pose_results, all_boxes)
                else:
                    write_record(fl, frame_index, frame_index/fps, None, None)

                frame_index += 1
                frame_valid, frame = cap.read()
                if frame_valid:
                    frame_r = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
            fl.close()
            cap.release()
    cv2.destroyAllWindows()

def main():
    vid_list = []
    vid_dir = './video_unseg'
    for x in os.listdir(vid_dir):
        if x.endswith(".avi"):
            vid_list.append(x)
    generate_sigs(vid_list, vid_dir, './sigs_ut')

if __name__ == '__main__':
    main()

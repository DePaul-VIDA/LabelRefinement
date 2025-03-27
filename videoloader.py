from os.path import exists
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from fileloader import load_people
from filedata import FileData

#from mmengine.registry import init_default_scope
#from mmpose.apis import inference_topdown

#from mmdet.apis import inference_mot, init_track_model
#from mmpose.evaluation.functional import nms
#from mmpose.apis import init_model as init_pose_estimator
#from mmpose.utils import adapt_mmdet_pipeline
#from mmdet.apis import inference_detector, init_detector
#from mmpose.registry import VISUALIZERS
import enum

class Label(enum.Enum):
    HAND_SHAKING = 0
    HUGGING = 1
    KICKING = 2
    POINTING = 3
    PUNCHING = 4
    PUSHING = 5
    BACKGROUND = -1

    def to_str(self, l):
        if l == self.HAND_SHAKING:
            return "Hand Shaking"
        elif l == self.HUGGING:
            return "Hugging"
        elif l == self.KICKING:
            return "Kicking"
        elif l == self.POINTING:
            return "Pointing"
        elif l == self.PUNCHING:
            return "Punching"
        elif l == self.PUSHING:
            return "Pushing"
        elif l == self.BACKGROUND:
            return "Background"

    def from_str(label):
        if 'HUGGING' in label:
            return Label.HUGGING
        elif 'HAND_SHAKING' in label:
            return Label.HAND_SHAKING
        elif'PUNCHING' in label:
            return Label.PUNCHING
        elif 'PUSHING' in label:
            return Label.PUSHING
        elif 'KICKING' in label:
            return Label.KICKING
        else:
            raise NotImplementedError


class LabelRange:
    def __init__(self, label, start, end):
        self.label = Label(label)
        self.start = start
        self.end = end

    def copy(self):
        return LabelRange(self.label, self.start, self.end)


class Video:
    def __init__(self, file, xml_file, length, annotation_list, label_entries):
        self.file_name = file
        self.xml_file = xml_file
        self.length = length
        self.annotation_list = annotation_list
        self.label_entries = label_entries

        fd = FileData()
        fd.file_name_full = file + '.csv'
        print(fd.file_name_full)
        self.tracked_persons, fd.frame_count, m, pf = load_people(fd.file_name_full, fd)
        for tp in self.tracked_persons.values():
            #tp.trim_ankles()
            tp.calc_signals()

    def get_boxes_at_frame(self, frame_index):
        boxes = []
        for tp in self.tracked_persons.values():
            index = frame_index - tp.start_frame
            if index > -1:
                boxes.append(tp.bounding_boxes[frame_index])
        return boxes


class VideoLoader:
    def __init__(self, file, device):
        self.video_list = []
        self.curr_frame = 0
        self.curr_img = None
        self.prev_img = None
        self.cap = None
        self.video = None
        self.load_new_video(file)
        self.use_pose = False
        self.colors = [(255, 0, 0), (125, 0, 125), (0, 255, 0), (0, 0, 255), (0, 125, 125), (125, 125, 0)]


        if self.use_pose:
            init_default_scope('mmdet')
            # build the model from a config file and a checkpoint file

            self.det_model = init_track_model(
                './ModelSetup/bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval.py',
                './ModelSetup/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth',
                None,
                None,
                device=device)

            # detector = init_detector('./ModelSetup/rtmdet_m_640-8xb32_coco-person.py', './ModelSetup/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth', device)
            self.det_model.cfg = adapt_mmdet_pipeline(self.det_model.cfg)
            self.pose_model = init_pose_estimator(
                './ModelSetup/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py',
                './ModelSetup/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
                device=device, cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

            self.pose_model.cfg.visualizer.radius = 3
            self.pose_model.cfg.visualizer.alpha = 0.8
            self.pose_model.cfg.visualizer.line_width = 1
            self.visualizer = VISUALIZERS.build(self.pose_model.cfg.visualizer)
            # the dataset_meta is loaded from the checkpoint and
            # then pass to the model in init_pose_estimator
            self.visualizer.set_dataset_meta(
                self.pose_model.dataset_meta, skeleton_style='mmpose')
        else:
            self.pose_model = None
            self.det_model = None
            self.visualizer = None


    def get_scaled_frame_at(self, index):
        if self.curr_frame == index:
            pass
        elif abs(index - self.curr_frame) > 10:
            self.move_to_image(index)
            self.read_image()
        elif self.curr_frame < index :
            while self.curr_frame < index - 1:
                self.cap.read()
                self.curr_frame += 1
            self.read_image()
        else:
            while self.curr_frame > index - 1:
                self.read_prev_image()
            self.read_image()
        #scaled = cv2.resize(self.curr_img, (800, 450), interpolation=cv2.INTER_CUBIC)
        #self.draw_people()
        self.curr_frame = index
        return self.curr_img

    def load_new_video(self, file):
        self.video_list = []
        self.curr_frame = 0
        self.curr_img = None
        self.prev_img = None
        file_no_ext = file.replace(".avi", "")
        self.cap = cv2.VideoCapture(file)
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        label_entries, annotation_list = self.__load_label_from_files(file_no_ext, 0, length)
        self.video = Video(file, file_no_ext, length, annotation_list, label_entries)

    def move_to_image(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 2)
        res, self.prev_img = self.cap.read()
        res, self.curr_img = self.cap.read()
        self.curr_frame = frame_number


    def get_img_arr(self, img):
        blue, green, red = cv2.split(img)
        img = cv2.merge((red, green, blue))
        return img

    def get_rgb_pose(self):
        if self.use_pose:
            self.__get_pose_results()
            img_pos = self.visualizer.get_image()
            return img_pos

        else:
            return self.curr_img

    def __get_pose_results(self):
        to_remove = []
        if self.use_pose:
            det_result = inference_mot(self.det_model, self.curr_img, frame_id=self.curr_frame, video_len=self.video.length)
            # pose result
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
                pose_results = inference_topdown(self.pose_model, self.curr_frame, bboxes_ps)

            for i in range(0, len(pose_results)):
                prob = pose_results[i]['bbox'][4]
                if prob < 0.8:
                    to_remove.append(i)

            if len(to_remove) > 0:
                to_remove.sort(reverse=True)
                for ti in to_remove:
                    del pose_results[ti]
            return pose_results
        return self.curr_img

    def __process_mmdet_results(self, mmtracking_results):

        person_results = []
        if self.use_pose:

            det_result = inference_mot(self.det_model, self.curr_img, frame_id=self.curr_frame, video_len=self.video.length)
            # pose result
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
            for track in bboxes_ps:
                person = {}
                person['track_id'] = int(track[0])
                person['bbox'] = track[1:]
                person['bbox'][2] = person['bbox'][2] - person['bbox'][0]
                person['bbox'][3] = person['bbox'][3] - person['bbox'][1]
                person_results.append(person)

        return person_results


    def scale_box(self, box):
        return [ x * 2 for x in box ]

    def draw_people(self):
        for tp in self.video.tracked_persons.values():
            i = int(self.curr_frame - tp.start_frame)
            l = len(tp.bounding_boxes)
            if 0 <= i < l:
                box = tp.bounding_boxes[i]
                box = self.scale_box(box)
                start_point = (int(box[0]), int(box[1]))
                start_point_offset = (int(box[0]), int(box[1]) - 20)

                end_point = (int(box[2]), int(box[3]))
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                self.curr_img = cv2.rectangle(self.curr_img, start_point, end_point, self.colors[int(tp.id)], thickness)
                cv2.putText(self.curr_img, 'Person ' + str(int(tp.id) + 1), start_point_offset, font, 1, self.colors[int(tp.id)], 2, cv2.LINE_AA)
        return self.curr_img

    def read_image(self):
        valid, self.curr_img = self.cap.read()
        self.curr_img = cv2.resize(self.curr_img, (640, 360), interpolation=cv2.INTER_CUBIC)
    def read_next_image(self):
        self.prev_img = self.curr_img
        if self.curr_frame <= self.video.length:
            self.read_image()

            self.curr_frame += 1
            #self.draw_people()


            #self.curr_img = cv2.resize(self.curr_img, (640, 360), interpolation=cv2.INTER_CUBIC)

            if self.use_pose:
                self.det_result = inference_mot(self.det_model, self.curr_img, frame_id=int(self.curr_frame), video_len=self.video.length)

            return self.curr_img
        return None

    def read_prev_image(self):
        if self.curr_frame > 0:
            next_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.curr_frame = next_frame - 1
            previous_frame = self.curr_frame - 1

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, previous_frame)
            self.read_image()

            #self.curr_img = cv2.resize(self.curr_img, (640, 360), interpolation=cv2.INTER_CUBIC)

            self.curr_frame += 1
            return self.curr_img
        return None

    def __load_label_from_files(self, file_path, start, end):
        out = []
        anns = open("ut-interaction.csv", "r")
        i = file_path.rfind('/')
        file_name = file_path[i+1:]
        anns.readline()
        anns.readline()
        lines = anns.readlines()
        for line in lines:
            vals = line.split(',')
            vid = vals[1].strip()
            if vid == file_name:
                out.append(LabelRange(int(vals[2].strip()), int(vals[3].strip()), int(vals[4].strip())))
        ann_list = []
        count = 0
        for l in out:
            while count < l.start:
                ann_list.append("Background")
                count += 1
            while count <= l.end:
                ann_list.append(l.label.to_str(l.label))
                count += 1
        while count <= end:
            ann_list.append("Background")
            count += 1

        return out, ann_list


    def __load_label_as_list(self, file_path):
        with open(file_path, 'r') as fp:
            lines = fp.read().splitlines()

        action, labels_entry = self.__get_action_entries(lines)
        return action, lines, labels_entry

    def __get_action_entries(self, ann_list):
        labels_entry = []
        i = 1

        current_label = ''
        start = 0
        end = 0
        for ann in ann_list:
            action = ann
            if current_label != ann:
                if current_label != '':
                    labels_entry.append(LabelRange(current_label, start, end))
                start = i
                current_label = ann
            end = i
            i += 1
        labels_entry.append(LabelRange(current_label, start, end))
        return action, labels_entry

    def __save_ann_list(self, file_path, ann_list):
        with open(file_path, 'w') as fp:
            fp.write('\n'.join(ann_list))




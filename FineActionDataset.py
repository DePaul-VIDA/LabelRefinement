import numpy
import cv2
from filedata import FileData
from fileloader import load_people
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import randint
#from signalextractor import generate_sigs
import json
from abc import abstractmethod
#from mmflow.apis import inference_model, init_model
import numpy as np
from random import randint
import os

IMAGE_WIDTH = 640

BACKGROUND = 0
HUG = 0
HAND_SHAKE = 0

list_labels = ['background', 'shake hands', 'hug']
BACKGROUND = 0
HAND_SHAKE = 1
HUG = 2
dict_labels = {'background':0, 'shake hands':1, 'hug':2}

class Sample:
    def __init__(self, sample_video, label, frame_num):
        self.sample_video = sample_video
        self.label = label
        self.frame_num = frame_num
        self.label_num = dict_labels[label]

    def set_flow_frame(self, flow_frame):
        self.flow_frame = flow_frame

    def set_flow_frame(self, sig_block):
        self.sig_block = sig_block


class SampleVideoSegment:
    def __init__(self, video_file, label, start_time, end_time, fps):
        self.video_file = video_file
        self.start_time = start_time
        self.end_time = end_time
        if start_time > end_time:
            self.start_time = end_time
            self.end_time = start_time

        self.label = label
        self.start_frame = int(round(self.start_time * fps, 0))
        self.end_frame = int(round(self.end_time * fps, 0))

    def get_frames(self):
        return self.end_frame - self.start_frame + 1

    def get_tIoU(self, start_frame, end_frame):
        union = float(max(end_frame, self.end_frame) - min(self.start_frame, start_frame))
        intersection = float(min(end_frame, self.end_frame) - max(self.start_frame, start_frame))
        return intersection / union


class SampleVideo:
    def __init__(self, video_file):
        self.segments = []
        self.video_file = video_file
        self.samples = []

    def add_segment(self, segment):
        self.segments.append(segment)

    def create_samples(self, end_vid_frame):
        current_frame = 1
        if len(self.segments) == 0:
            return

        cap = cv2.VideoCapture('./fineaction/' + self.video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        for seg in self.segments:
            while current_frame < seg.start_frame and current_frame < total_frames - 2:
                s = Sample(self, 'background', current_frame)
                self.samples.append(s)
                current_frame += 1

            while current_frame <= seg.end_frame and current_frame < total_frames - 2:
                s = Sample(self, seg.label, current_frame)
                self.samples.append(s)
                current_frame += 1

        while current_frame <= end_vid_frame-2 and current_frame < total_frames - 2:
            s = Sample(self, 'background', current_frame)
            self.samples.append(s)
            current_frame += 1

    def get_tIoU(self, start_frame, end_frame, curr):
        max_tIoU = 0.0
        for seg in self.segments:
            if dict_labels[seg.label] == curr:
                curr_tIoU = seg.get_tIoU(start_frame, end_frame)
            else:
                curr_tIoU = 0.0
            max_tIoU = max(max_tIoU, curr_tIoU)
        return max_tIoU


class FineActionDatasetLoader():

    def __init__(self, file_name):
        self.file_name = file_name
        self.training_videos, self.testing_videos, self.all_vids = self.load_file()

        #generate_sigs(self.all_vids, './videos', './sigs')

    def parse_samples(self, data):
        vid_list = []
        training = []
        testing = []
        self.ground_truth_counts = {1: 0, 2: 0}
        for vid in data['database'].values():
            file = vid['filename']
            if not os.path.exists("./fineaction/" + file):
                continue

            fps = vid['fps']
            frame_count = vid['actual_frame_num']
            hugs = 0
            hands = 0
            sv = SampleVideo(file)

            for ann in vid['annotations']:
                label = ann['label']
                if label not in list_labels:
                    continue
                svs = SampleVideoSegment(file, label, ann['segment'][0], ann['segment'][1], fps)
                sv.add_segment(svs)
                if file not in vid_list:
                    vid_list.append(file)
                if vid['subset'] == 'training':
                    if label == 'hug':
                        hugs += svs.get_frames()
                    if label == 'shake hands':
                        hands += svs.get_frames()
                elif vid['subset'] == 'validation':
                    if label == 'hug':
                        self.ground_truth_counts[HUG] += 1
                    if label == 'shake hands':
                        self.ground_truth_counts[HAND_SHAKE] += 1

            if len(sv.segments) > 0:
                if vid['subset'] == 'training':
                    training.append(sv)
                elif vid['subset'] == 'validation':
                    testing.append(sv)
                sv.create_samples(frame_count)

        return training, testing, vid_list

    def load_file(self):
        f = open(self.file_name)
        data = json.load(f)
        f.close()
        return self.parse_samples(data)

class FineActDataset(Dataset):
    def __init__(self, video_list, window_size, under_sample, maxes):
        self.sampleVids = video_list
        self.labels = []
        self.samples = []
        self.fileData = {}
        self.window_size = window_size
        self.load_samples_from_video(video_list)
        self.people_maxes = maxes

        counts = self.getLabelCounts()
        if under_sample:
            self.undersample(counts)
        self.n_samples = np.shape(self.samples)[0]

        self.y_data = torch.from_numpy(np.asarray(self.labels, dtype=numpy.integer))  # size [n_samples, 1]

    def load_samples_from_video(self, sample_vid):
        for vid in sample_vid:
            file_name = vid.video_file
            for s in vid.samples:
                self.samples.append([file_name, s.frame_num])
                self.labels.append(s.label_num)

    def getAlternateOrder(self, tp):
        tp.calc_signals()
        joints = []
        joints.append(tp.head)
        joints.append(tp.left_wrist)
        joints.append(tp.right_wrist)
        joints.append(tp.right_ankle)
        joints.append(tp.left_ankle)
        joints.append(tp.left_knee)
        joints.append(tp.right_knee)
        joints.append(tp.left_elbow)
        joints.append(tp.right_elbow)
        return joints


    def load_signals_from_video(self, file_name):
        fd = FileData()
        tracked_persons, fd.frame_count, max_person_count, people_frame = load_people('./sigs/' + file_name + ".csv", fd)
        self.fileData[file_name] = [fd, tracked_persons]
        return max_person_count, people_frame

    def getPeopleAtFrame(self, tracked_persons, i):
        tp_dict = {}
        for tp in tracked_persons.values():
            if tp.start_frame > i or tp.end_frame < i:
                continue
            tp_dict[tp] = self.getAlternateOrder(tp)

        peops = 0
        if len(tp_dict.values()) == 0:
            peops = 0
            signals = np.zeros((1, 9, 1), dtype=float)
        elif len(tp_dict.values()) < self.people_maxes[0]+1:
            peops = self.people_maxes[0]
        elif len(tp_dict.values()) < self.people_maxes[1]+1:
            peops = self.people_maxes[1]
        else:
            peops = 25
        return peops, tp_dict


    def getLabelCounts(self):
        counts = {BACKGROUND:0, HAND_SHAKE:0, HUG:0}
        for l in self.labels:
            counts[l] += 1

        print("Backgorund: " + str(counts[BACKGROUND]))
        print("Hand Shake: " + str(counts[HAND_SHAKE]))
        print("Hug: " + str(counts[HUG]))
        return counts

    def undersample(self, counts):
        #to_value = counts[min(counts, key=counts.get)]
        to_value = min(counts[HAND_SHAKE], counts[HUG])

        reduce = []
        reduce.append([BACKGROUND, counts[BACKGROUND] - to_value])
        reduce.append([HAND_SHAKE, counts[HAND_SHAKE] - to_value])
        reduce.append([HUG, counts[HUG] - to_value])

        while (reduce[0][1] > 0 or reduce[1][1] > 0 or reduce[2][1] > 0):
            i = randint(0, len(self.samples)-1)
            #print("List size is: " + str(len(self.labels)) + " i is: " + str(i))
            for j in range(0, 3):
                if self.labels[i] == reduce[j][0] and reduce[j][1] > 0:
                    del self.labels[i]
                    del self.samples[i]
                    reduce[j][1] -= 1
                    break

    def __len__(self):
        self.n_samples

    def __getitem__(self, index):
        pass


class FineActionDatasetSignals(FineActDataset):
    def __init__(self, vids, device, window_size, under_sample, maxes):
        super(FineActionDatasetSignals, self).__init__(vids,  window_size, under_sample, maxes)
        self.vids = vids
        curr_max = 0
        bins_shake = {}
        bins_hug = {}
        bins_background = {}
        bins = {}

        for i in range(0, 26):
           bins[i] = 0
           bins_hug[i] = 0
           bins_background[i] = 0
           bins_shake[i] = 0
        for vid in vids:
            if not vid.video_file in self.fileData.keys():
               count, list_people = self.load_signals_from_video(vid.video_file)
               curr_max = max(curr_max, count)
               for i in range(0, len(list_people)):
                   bins[list_people[i]] += 1
               for sam in vid.samples:
                   if sam.frame_num <= len(list_people):
                       if sam.label_num == BACKGROUND:
                           bins_background[list_people[sam.frame_num - 1]] += 1
                       elif sam.label_num == HUG:
                           bins_hug[list_people[sam.frame_num - 1]] += 1
                       elif sam.label_num == HAND_SHAKE:
                           bins_shake[list_people[sam.frame_num - 1]] += 1

      #  print("Totals: ")
      #  for i in range(0, 26):
      #      print(str(i) + ": " + str(bins[i]))

      #  print("Background: ")
     #   for i in range(0, 26):
     #       print(str(i) + ": " + str(bins_background[i]))

      #  print("Hug: ")
      #  for i in range(0, 26):
      #      print(str(i) + ": " + str(bins_hug[i]))

       # print("Shake: ")
      #  for i in range(0, 26):
        #    print(str(i) + ": " + str(bins_shake[i]))


        self.max_people = curr_max


    def getSignals(self, file_name, i):
        fd, tracked_persons = self.fileData[file_name]
        max_people_per_panel = self.max_people

        peops, tp_dict = self.getPeopleAtFrame(tracked_persons, i)
        hw = int((self.window_size - 1) / 2)
        signals = np.zeros((1, 9, peops * (self.window_size + 1)), dtype=float)
        person_count = 0
        offset = 0
        for tp in tracked_persons.values():
            if tp.start_frame > i or tp.end_frame < i:
                continue

            joints = tp_dict[tp]

            for j in range(0, len(joints)):
                offset = 0
                for l in range(i - hw + 2, i + hw + 3):  # signal data in this set is shifted by 2
                    for ss in joints[j].subsigs:
                        if ss.end_frame > l >= ss.start_frame:
                            signals[0][j][offset + (self.window_size * person_count)] = ss.speed_smooth[l - ss.start_frame]
                            break
                    offset += 1
            for j in range(0, len(joints)):
                signals[0][j][offset + (self.window_size * person_count)] = 0
            person_count += 1

            if person_count >= max_people_per_panel:
                break

        return signals, peops


    def __getitem__(self, index):
        file_name, i = self.samples[index]
        fd, tracked_persons = self.fileData[file_name]

        signals, peops = self.getSignals(file_name, i)
        return torch.from_numpy(signals).type(torch.FloatTensor), self.y_data[index], [file_name, i, fd.frame_count, peops]


    def __len__(self):
        return self.n_samples

class FineActionDatasetFrames(FineActDataset):

     def __init__(self, video_list, device, window_size, under_sample):
         super(FineActionDatasetFrames, self).__init__(video_list,  window_size, under_sample, [2, 8])
         config_file = 'ModelSetup/mmflow/Config/configs/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.py'
         checkpoint_file = 'ModelSetup/mmflow//checkpoints/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'
         self.flow_model = init_model(config_file, checkpoint_file, device=device)
         self.cur_sam = 0
     
     def getNextFrame(self, file_name, index):
         self.cur_sam += 1
         count = 0
         cap = cv2.VideoCapture('./fineaction/' + file_name)
         cap.set(cv2.CAP_PROP_POS_FRAMES, index)
         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
         #for i in range(0, index - 1):
          #  count += 1
           # cap.grab()
         
         #print("video: " + file_name + " frame: " + str(index) + " curr: " + str(count) + " total: " + str(total_frames) + " -- " + str(self.cur_sam) + " of " + str(self.n_samples))
         valid1, currRGB = cap.read()
         valid2, nextRGB = cap.read()
         currRGB = cv2.resize(currRGB, (320, 180), interpolation=cv2.INTER_CUBIC)
         nextRGB = cv2.resize(nextRGB, (320, 180), interpolation=cv2.INTER_CUBIC)

         next_flow_result = inference_model(self.flow_model, currRGB, nextRGB)  # [720, 1280, 2]
         return next_flow_result, total_frames


     def __getitem__(self, index):
         file_name, i = self.samples[index]
         next_flow_result, total_frames = self.getNextFrame(file_name, i)
         x = numpy.transpose(next_flow_result, (2, 0, 1))
         return torch.from_numpy(x).type(torch.FloatTensor), self.y_data[index], [file_name, i, total_frames, -1]

     def __len__(self):
         return self.n_samples


class FineActionDatasetPoints(FineActDataset):
    def __init__(self, vids, device, window_size, under_sample, maxes):
        super(FineActionDatasetPoints, self).__init__(vids,  window_size, under_sample, maxes)
        self.vids = vids
        for vid in vids:
            if not vid.video_file in self.fileData.keys():
               count, list_people = self.load_signals_from_video(vid.video_file)

        self.max_people = 25

    def getPoints(self, file_name, i):
        fd, tracked_persons = self.fileData[file_name]
        max_people_per_panel = self.max_people

        peops, tp_dict = self.getPeopleAtFrame(tracked_persons, i)

        points = np.zeros((1, 9 * peops), dtype=float)
        person_count = 0
        for tp in tracked_persons.values():
            if tp.start_frame > i or tp.end_frame < i:
                continue
            offset = 0

            joints = tp_dict[tp]
            for j in range(0, len(joints)):
                for ss in joints[j].subsigs:
                    if ss.end_frame > i >= ss.start_frame:
                        points[0][offset + 9*person_count] = ss.speed_smooth[i - ss.start_frame]
                        break
                offset += 1
            person_count += 1

            if person_count >= max_people_per_panel:
                break

        return points, peops


    def __getitem__(self, index):
        file_name, i = self.samples[index]
        fd, tracked_persons = self.fileData[file_name]

        points, peops = self.getPoints(file_name, i)
        return torch.from_numpy(points).type(torch.FloatTensor), self.y_data[index], [file_name, i, fd.frame_count, peops]


    def __len__(self):
        return self.n_samples

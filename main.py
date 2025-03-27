import torch
import os
from sample import create_samples
from tkinter import *
from tkinter.ttk import *
# from videoloader import VideoLoader, LabelRange, Label as vlLabel
from videoloader import VideoLoader, Label

from PIL import Image, ImageTk
from numpy import array
import matplotlib
from autorevise_template import AutoreviseTemplate
import cv2
import shutil

# from generatedata import inference
from fileloader import load_people
from filedata import FileData

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

vl = None
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(dev)
playing = False

fr_orig_start = 1137
fr_orig_end = 1205
fr_rev_start = 1151
fr_rev_end = 1186
Activity = "Handshaking"
key_frame_list = [fr_orig_start, fr_orig_end, fr_rev_start, fr_rev_end]
len_fr = fr_orig_end - fr_orig_start
expand = len_fr * .15
fr_start = fr_orig_start - int(expand)
fr_end = fr_orig_end + int(expand)

root = Tk()
# root.geometry('1350x1050')
# root.geometry('2200x1050')
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
print('height = ' + str(screen_height) + ' width = ' + str(screen_width))
root.geometry(f"{screen_width}x{screen_height}+0+0")
root.title("Annotation Viewer")

values = {"Image": 1,
          "Pose": 2,
          "Flow/Pose": 3}

video_str = StringVar(root, "./sigs_ut/seq15.avi")
video_path_input = Entry(root, textvariable=video_str, width=80)
video_path_input.place(relx=0.055, rely=0.018)

Label(root, text="Go to frame").place(relx=0.36, rely=0.017)
video_frame_str = StringVar(root, str(fr_start))
video_frame_input = Entry(root, text=video_frame_str, width=25)
video_frame_input.place(relx=0.40, rely=.018)

imgtk = None
image_disp = Label(root)
image_disp.place(relx=0.02, rely=0.040)
fps_str = StringVar(root, "15")
fps_input = Entry(root, text=fps_str, width=20)
fps_input.place(relx=0.23, rely=0.355)
image_type = IntVar(root, 1)
signal_type = IntVar(root, 1)
person_type = []
for i in range(0, 16):
    person_type.append(IntVar(root, 0))
annotation = Label(root)
annotation.place(relx=0.316, rely=0.355)
frame_num = Label(root)
frame_num.place(relx=0.364, rely=0.355)

fig = plt.figure(figsize=(0.98 * screen_width / 100, .5 * screen_height / 100), dpi=100)
plt.ion()
canvas = FigureCanvasTkAgg(fig, master=root)
plot_widget = canvas.get_tk_widget()
plot_widget.place(relx=0.01, rely=0.443)

Label(root, text="y limit").place(relx=0.021, rely=0.415)
y_lim_str = StringVar(root, "110")
y_lim_input = Entry(root, text=y_lim_str, width=20)
y_lim_input.place(relx=0.053, rely=0.415)

rw_bool = BooleanVar(root, True)
lw_bool = BooleanVar(root, True)
ra_bool = BooleanVar(root, True)
la_bool = BooleanVar(root, True)
re_bool = BooleanVar(root, False)
le_bool = BooleanVar(root, False)
rk_bool = BooleanVar(root, False)
lk_bool = BooleanVar(root, False)

graph_person_values = {"Person 1": 1,
                       "Person 2": 2,
                       "Person 3": 3,
                       "Person 4": 4,
                       "Person 5": 5,
                       "Person 6": 6,
                       "Person 7": 7,
                       "Person 8": 8,
                       "Person 9": 9,
                       "Person 10": 10,
                       "Person 11": 11,
                       "Person 12": 12,
                       "Person 13": 13,
                       "Person 14": 14,
                       "Person 15": 15,
                       "All": 16}

autorevise_template = AutoreviseTemplate()


def get_loader():
    global video_str
    file = video_str.get()
    global vl
    vl.load_new_video(file)
    vl.read_next_image()

    frame = int(video_frame_str.get())
    vl.move_to_image(frame)

    load_image()
    get_next()


def get_next():
    global vl
    vl.read_next_image()
    load_image()


def get_prev():
    global vl
    vl.read_prev_image()
    load_image()


def play():
    global playing
    if playing:
        playing = False
    else:
        playing = True
    do_play()


def do_play():
    global vl
    global playing
    global fps_str
    if playing:
        fps = int(fps_str.get())
        get_next()
        delay = int(1000 / fps)
        if vl.curr_frame in key_frame_list:
            delay = delay * 25
        if vl.curr_frame > fr_end:
            playing = False

        image_disp.after(delay, do_play)


def load_image():
    global image_disp
    global imgtk
    global annotation
    global image_type

    orig_im = None
    if image_type.get() == 1:
        orig_im = vl.curr_img
    elif image_type.get() == 2:
        orig_im = vl.get_rgb_pose()
    elif image_type.get() == 3:
        orig_im = vl.get_rgb_pose()
    orig_im = vl.get_rgb_pose()

    img = vl.get_img_arr(orig_im)
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    image_disp.config(image=imgtk)
    index = int(vl.curr_frame - 1)
    if index < len(vl.video.annotation_list):
        ann = vl.video.annotation_list[index]
    else:
        ann = ''
    ann = ''
    annotation.config(text=ann)
    frame_num.config(text=str(vl.curr_frame))
    plot_signals()


def plot_signals():
    speed = True
    label_entries = vl.video.label_entries
    right_wrist = rw_bool.get()
    left_wrist = lw_bool.get()
    right_ankle = ra_bool.get()
    left_ankle = la_bool.get()
    right_elbow = re_bool.get()
    left_elbow = le_bool.get()
    right_knee = rk_bool.get()
    left_knee = lk_bool.get()
    y_lim = int(y_lim_str.get())
    max = y_lim
    i = 1
    plt.clf()
    legend_list = []
    legend_string = []
    ls = ''

    colors = ['blue', 'orange', 'red', 'green', 'gray']
    index = 0
    ls = 'Annotated Action Start/End'
    l1 = plt.vlines(x=fr_end + 300, ymin=0, ymax=max,
                    colors='purple',
                    ls='--',
                    label=ls)

    c = 100
    while c < vl.video.length:
        #    l1 = plt.vlines(x=c, ymin=0, ymax=max,
        #                   colors='orange',
        #                  ls='--',
        #                 label=ls)
        c += 100
    plt.ylim(0, y_lim)
    plt.axis([fr_start - 5, fr_end + 40, 0, y_lim])

    zone = int((fr_end - fr_start) / 5)
    x_s = fr_start + zone
    ls = 'Zones'
    for b in range(0, 5):
        l1 = plt.vlines(x=x_s, ymin=0, ymax=max,
                        colors='purple',
                        ls='--',
                        label=ls)
        x_s += zone
    legend_list.append(l1)
    legend_string.append(ls)

    ls = 'Human Start/End'
    l1 = plt.vlines(x=fr_orig_start, ymin=0, ymax=max,
                    colors='blue',
                    ls='--',
                    label=ls)

    l1 = plt.vlines(x=fr_orig_end, ymin=0, ymax=max,
                    colors='blue',
                    ls='--',
                    label=ls)
    legend_list.append(l1)
    legend_string.append(ls)

    ls = 'Refined Start/End'
    l1 = plt.vlines(x=fr_rev_start, ymin=0, ymax=max,
                    colors='green',
                    ls='--',
                    label=ls)

    l1 = plt.vlines(x=fr_rev_end, ymin=0, ymax=max,
                    colors='green',
                    ls='--',
                    label=ls)
    legend_list.append(l1)
    legend_string.append(ls)

    # ls = 'Annotated Action Start/End'
    #    l1 = plt.vlines(x=vl.video.label_entries[2].start, ymin=0, ymax=max, colors='blue', ls='--', label=ls)
    #    l1 = plt.vlines(x=vl.video.label_entries[2].end, ymin=0, ymax=max, colors='blue', ls='--', label=ls)
    #    legend_list.append(l1)
    #    legend_string.append(ls)

    for tp in vl.video.tracked_persons.values():

        id = int(float(tp.id))
        if id > 14 or person_type[id].get() or person_type[15].get():
            # legend_list.append(l1)
            # legend_string.append(ls)
            pre = 'P' + str(i) + ' - '

            if right_wrist:
                ls = 'RWrist'
                do_plot_sig(tp.right_wrist, ls, speed, legend_list, legend_string, pre)
            if left_wrist:
                ls = 'LWrist'
                do_plot_sig(tp.left_wrist, ls, speed, legend_list, legend_string, pre)
            if left_ankle:
                ls = 'LAnkle'
                do_plot_sig(tp.left_ankle, ls, speed, legend_list, legend_string, pre)
            if right_ankle:
                ls = 'RAnkle'
                do_plot_sig(tp.right_ankle, ls, speed, legend_list, legend_string, pre)
            if left_elbow:
                ls = 'LElbow'
                do_plot_sig(tp.left_elbow, ls, speed, legend_list, legend_string, pre)
            if right_elbow:
                ls = 'RElbow'
                do_plot_sig(tp.right_elbow, ls, speed, legend_list, legend_string, pre)
            if left_knee:
                ls = 'LKnee'
                do_plot_sig(tp.left_knee, ls, speed, legend_list, legend_string, pre)
            if right_knee:
                ls = 'RKnee'
                do_plot_sig(tp.right_knee, ls, speed, legend_list, legend_string, pre)

            vid = vl.video.file_name.replace('D:\\ShakeFive2\\', '').replace('.mp4', '')
            if speed:
                plt.title('Person speed signals, wrists and ankles: ' + Activity + " Activity")
                plt.ylabel('Speed px/33.33ms')
            else:
                plt.ylabel('Acceleration px/33.33ms^2')
                plt.title('Person acceleration signals, wrists and ankles')
        i += 1

    plt.xlabel('Frame number')
    plt.legend(legend_list, legend_string)


def do_plot_sig(joint, ls, speed, legend_list, legend_string, pre):
    count = 1
    for s in joint.subsigs:
        point_to_plot = int(vl.video.length)
        # point_to_plot = int(vl.curr_frame)
        if s.start_frame > point_to_plot:
            continue
        if s.end_frame < point_to_plot + 3:
            point_to_plot = s.end_frame - 3
        X = range(s.start_frame, point_to_plot + 3)
        if s.end_frame >= fr_start and s.start_frame <= fr_end:

            length = point_to_plot + 1 - s.start_frame
            if point_to_plot > s.end_frame:
                length = s.end_frame + 1 - s.start_frame
                X = s.x_axis()

            a = array(s.speed_smooth[:length])
            if len(a) < len(X):
                X = range(s.start_frame, s.start_frame + len(a))
            nls = ls + '-' + str(count)
            if speed:
                print(str(a.shape) + " -- " + str(len(X)) + " Current Frame: " + str(vl.curr_frame))
                l, = plt.plot(X, s.speed_smooth[:length], label=nls)
            else:
                l, = plt.plot(X, s.accel_smooth[:length], label=nls)
            legend_list.append(l)
            legend_string.append(pre + nls)
            count += 1


def mainGUI():
    y_val = 0.355
    i = 0
    global image_type
    Label(root, text="Video Path:").place(relx=0.021, rely=0.017)

    Button(root, text='Load', command=lambda: get_loader()).place(relx=0.31, rely=0.017)

    Button(root, text='Back', command=lambda: get_prev()).place(relx=0.020, rely=y_val)
    Button(root, text='Play', command=lambda: play()).place(relx=0.065, rely=y_val)
    Button(root, text='Forward', command=lambda: get_next()).place(relx=.11, rely=y_val)

    Label(root, text="FPS:").place(relx=.21, rely=y_val)

    y_val = 0.415
    Checkbutton(root, text='Right Wrist', variable=rw_bool, onvalue=True, offvalue=False, command=plot_signals).place(
        relx=0.234, rely=y_val)
    Checkbutton(root, text='Left Wrist', variable=lw_bool, onvalue=True, offvalue=False, command=plot_signals).place(
        relx=0.286, rely=y_val)
    Checkbutton(root, text='Right Ankle', variable=ra_bool, onvalue=True, offvalue=False, command=plot_signals).place(
        relx=0.338, rely=y_val)
    Checkbutton(root, text='Left Ankle', variable=la_bool, onvalue=True, offvalue=False, command=plot_signals).place(
        relx=0.390, rely=y_val)

    i += 1
    i = 0
    y_val -= 0.33

    for (text, value) in graph_person_values.items():
        Checkbutton(root, text=text, variable=person_type[i], onvalue=True, offvalue=False, command=plot_signals).place(
            relx=0.360, rely=y_val + i * 0.016)
        i += 1

    global video_str
    global vl
    file = video_str.get()
    vl = VideoLoader(file, device)
    get_loader()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


def close():
    root.quit()


def main():
    samples_path = "D:\\ShakeFive2Samples\\"
    videos_dir = "D:\\ShakeFive2\\"

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    index = 0
    with torch.no_grad():
        samples, index = create_samples(videos_dir, samples_path, dev)

    print('Done!!!')


def load_files():
    for file in os.listdir(dir):
        # check only text files
        if file.endswith('.csv'):
            fd = FileData()
            fd.file_name_full = dir + file
            tracked_persons, frame_count = load_people(fd.file_name_full, fd)
            print(fd.file_name_full)


def copy_list(orig):
    list = []
    for lr in orig:
        if lr is None:
            list.append(None)
        else:
            list.append(lr.copy())
    return list


def run_list(action_type, video_list, fl1, fl2, fl3, fl4, fl5):
    newDict = {}
    with open('d:\\videoaction.csv', 'r') as f:
        for line in f:
            splitLine = line.split(',')
            newDict[splitLine[0]] = splitLine[1:]

    fl1.write(action_type + "\n")
    fl2.write(action_type + "\n")
    fl3.write(action_type + "\n")

    vl = VideoLoader(video_list[0], device)
    AR_list = []
    label_list = []
    for file in video_list:
        print(file)
        vl.load_new_video(file)
        label = copy_list(vl.video.label_entries)
        label_list.append(label)
        AR = copy_list(autorevise(file))
        AR_list.append(AR)
        if label[0].label == vlLabel.APPROACH:
            approach = 0
            action = 1
            leave = 2
            stand2 = 3
        elif label[1].label == vlLabel.APPROACH:
            approach = 1
            action = 2
            leave = 3
            stand2 = 4
        else:
            print("ERROR!\n\n\n\\n")

        fl1.write(file + ", " + str(label[approach].start) + ", " + str(AR[1].start) + "\n")
        fl2.write(file + ", " + str(label[action].start) + ", " + str(AR[2].start) + "\n")
        fl3.write(file + ", " + str(label[leave].start) + ", " + str(AR[3].start) + "\n")
        fl5.write(file + ", " + str(label[action].label).replace("Label.", "") + ", " + str(
            label[approach].start) + ", " + str(label[approach].end) + ", ")
        fl5.write(
            str(label[action].start) + ", " + str(label[action].end) + ", " + str(label[leave].start) + ", " + str(
                label[leave].end) + ", ")
        if len(label) == stand2:
            fl5.write("-1, -1, ")
        else:
            fl5.write(str(label[stand2].start) + ", " + str(label[stand2].end) + ", ")
        fl5.write(
            str(AR[1].start) + ", " + str(AR[1].end) + ", " + str(AR[2].start) + ", " + str(AR[2].end) + ", " + str(
                AR[3].start) + ", " + str(AR[3].end) + ", ")
        if AR[stand2] is None:
            fl5.write("-1, -1\n")
        else:
            fl5.write(str(AR[stand2].start) + ", " + str(AR[stand2].end) + "\n")

    fl1.write("\n\n\n")
    fl2.write("\n\n\n")
    fl3.write("\n\n\n")
    fl1.flush()
    fl2.flush()
    fl3.flush()
    fl4.flush()
    fl5.flush()


def load_vids():
    video_list = []
    with open("D:\\run_vids.txt", 'r') as fl:
        for line in fl:
            video_list.append("D:\\ShakeFive2\\" + line.strip())
    return video_list


def run_all_lists():
    fl1 = open("D:\\output_approach.csv", 'w')
    fl2 = open("D:\\output_action.csv", 'w')
    fl3 = open("D:\\output_leave.csv", 'w')
    fl4 = open("D:\\output_intersection.csv", 'w')
    fl5 = open("D:\\autorevise_results.csv", 'w')

    run_list("unknown", load_vids(), fl1, fl2, fl3, fl4, fl5)
    fl1.close()
    fl2.close()
    fl3.close()
    fl4.close()
    fl5.close()


def get_video_stats(path, results):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            cap = cv2.VideoCapture("D:\\ShakeFive2\\" + line)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fr = int(cap.get(cv2.CAP_PROP_FPS))
            results.write(line + "," + str(length) + "," + str(fr) + "\n")


def write_stats():
    output = open("D:\\output_vid.csv", 'w')

    output.write("Hug\n")
    get_video_stats("D:\\ShakeFive2\\hug.txt", output)
    output.write("\n\nHandshake\n")
    get_video_stats("D:\\ShakeFive2\\hand_shake.txt", output)
    output.write("\n\nHigh Five\n")
    get_video_stats("D:\\ShakeFive2\\high_five.txt", output)
    output.write("\n\nFist Bump\n")
    get_video_stats("D:\\ShakeFive2\\fist_bump.txt", output)
    output.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    dir = "D:\\ShakeFive2\\"
    # inference(dir)
    # load_files()
    # write_stats()

    # run_all_lists()
    # autorevise_template.run_ut_list()
    mainGUI()







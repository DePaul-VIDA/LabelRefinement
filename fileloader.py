
from trackedperson import TrackedPerson
from pathlib import Path

frame_x2 = 640
frame_y2 = 360

def is_prob(value):
    return value > 0.90


def append_keypoint(s, values, person_headers, offset, i, count, line_headers):
    s.x.append(float(values[i * person_headers + offset + line_headers]))
    s.y.append(float(values[i * person_headers + offset + 1 + line_headers]))
    s.prob.append(float(values[i * person_headers + offset + 2 + line_headers]))
#    s.u.append(float(values[i * person_headers + offset + 5 + line_headers]))
#    s.v.append(float(values[i * person_headers + offset + 6 + line_headers]))
    s.end_frame = count

def get_bounding_box(values, person_headers, i, line_headers):
    x2 = float(values[i * person_headers + 3 + line_headers])
    y2 = float(values[i * person_headers + 4 + line_headers])
    bbox = [float(values[i * person_headers + 1 + line_headers]),
                 float(values[i * person_headers + 2 + line_headers]), x2, y2]
    result = True
    if x2 >= frame_x2 or y2 >= frame_y2:
        result = False

    return bbox, result

def load_people(file_name, fd):

    my_file = Path(file_name)
    if not my_file.is_file():
        return [], 0, 0, 0
    f = open(file_name, newline='')
    f.readline()
    f.readline()
    line_headers = 3
    person_count_index = 2
    person_id_offset = 0
    person_bb_offset = 1
    left_wrist_x_offset = 33
    right_wrist_x_offset = 36
    left_elbow_x_offset = 27
    right_elbow_x_offset = 30

    left_ankle_x_offset = 51
    right_ankle_x_offset = 54
    left_knee_x_offset = 45
    right_knee_x_offset = 48
    head_x_offset = 6
    person_headers = 57

    people_frame = []

    done = False
    tracked_persons = {}
    count = 2
    split = False
    curr_max = 0
    while not done:
        line = f.readline()
        person = None

        if line is None or len(line) == 0:
            done = True
            continue

        values = line.strip().split(',')
        le = len(values)
        if le == 1:
            return tracked_persons, count
        elif le == 3:
            pc = 0
        else:
            pc = int(values[person_count_index])
        if pc > 25:
            pc = 25
        count += 1
        fd.frame_ref[count] = []
        curr_max = max(curr_max, pc)
        people_frame.append(pc)
        for i in range(0, pc):
            id = values[i * person_headers + person_id_offset + line_headers]
            id = id.strip()
            if not (id in tracked_persons.keys()):
                tracked_persons[id] = TrackedPerson()
                person = tracked_persons[id]
                person.start_frame = count
                person.end_frame = count-1
                person.id = id
            person = tracked_persons[id]
            person.end_frame = count
            fd.frame_ref[count].append(person)

            bbox, result = get_bounding_box(values, person_headers, i, line_headers)
            person.bounding_boxes.append(bbox)
            append_keypoint(person.left_wrist.get_create_subsig(count, True), values, person_headers, left_wrist_x_offset, i, count, line_headers)
            append_keypoint(person.right_wrist.get_create_subsig(count, True), values, person_headers, right_wrist_x_offset, i, count, line_headers)
            append_keypoint(person.left_ankle.get_create_subsig(count, result), values, person_headers, left_ankle_x_offset, i, count, line_headers)
            append_keypoint(person.right_ankle.get_create_subsig(count, result), values, person_headers, right_ankle_x_offset, i, count, line_headers)

            append_keypoint(person.left_elbow.get_create_subsig(count, result), values, person_headers, left_elbow_x_offset, i, count, line_headers)
            append_keypoint(person.right_elbow.get_create_subsig(count, result), values, person_headers, right_elbow_x_offset, i, count, line_headers)
            append_keypoint(person.left_knee.get_create_subsig(count, result), values, person_headers, left_knee_x_offset, i, count, line_headers)
            append_keypoint(person.right_knee.get_create_subsig(count, result), values, person_headers, right_knee_x_offset, i, count, line_headers)
            append_keypoint(person.head.get_create_subsig(count, result), values, person_headers, head_x_offset, i, count, line_headers)

    return tracked_persons, count, curr_max, people_frame

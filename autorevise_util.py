import numpy

move_threshold = 1
sortof_move_threshold = 0.5

def joints_moving_together(joint_set, length, start, is_moving_threshold):
    speed_at = []
    moving_together = numpy.zeros(length)
    sortof_moving_together = numpy.zeros(length)
    moving_together_inc_low = numpy.zeros(length)
    for i in range(0, length):
        speed_at.append([])
        for joint in joint_set:
            sigs = joint.get_subsig_at_range(i+start, i+start)
            for s in sigs:
                speed_at[i].append(s.speed_smooth[start+i-s.start_frame])
        moving_together[i], sortof_moving_together[i], moving_together_inc_low[i] = is_moving_together(speed_at[i], is_moving_threshold)
    segs = find_segments(moving_together_inc_low, 1)
    segs = remove_all_low(segs, moving_together)


    return segs
    #join_segs(mt_segs, find_segments(moving_together_inc_low))
    #, find_segments(sortof_moving_together)

def find_segments(vect, look_for):
    count = 0
    s = 0
    sections = []
    for i in range(0, len(vect)):
        if vect[i] == look_for:
            if count == 0:
                s = i + 1
            count += 1
        else:
            if count != 0:
                if s > 1:
                    sections.append([s, i])
                count = 0
    return sections

def is_moving_together(speeds, is_moving_threshold):
    if len(speeds) == 0:
        return 0, 0, 0
    a = numpy.average(speeds)
    d = sum([abs(i - a) for i in speeds])
    mt = 1 if move_threshold > d and a > is_moving_threshold else 0
    mtls = 1 if move_threshold > d else 0
    smt = 1 if sortof_move_threshold > d and a > is_moving_threshold else 0
    return mt, smt, mtls

def remove_all_low(segs, moving_together):
    res = []

    for s in segs:
        for i in range(s[0], s[1]):
            if moving_together[i] == 1:
                res.append(s)
                break
    return res

def connected_joint_close(section, joint_set):
    contact_thres = 1.5
    start = -1
    prev = 999999.0
    end = -1
    for i in range(section[0], section[1]):

        p1 = joint_set[0].get_subsig_at_range(i, i)
        p2 = joint_set[1].get_subsig_at_range(i, i)
        if len(p1) == 0 or len(p2) == 0:
            start = i
            break
        curr = abs(p1[0].x[i - p1[0].start_frame] - p2[0].x[i - p2[0].start_frame])
        if abs(prev - curr) < contact_thres:
            start = i - 1
            break
        prev = curr

    prev = 999999.0
    contact_range_thres = 25

    for i in reversed(range(section[0], section[1] + 1)):

        p1 = joint_set[0].get_subsig_at_range(i, i)
        p2 = joint_set[1].get_subsig_at_range(i, i)
        if len(p1) == 0 or len(p2) == 0:
            end = i
            break
        curr = abs(p1[0].x[i - p1[0].start_frame] - p2[0].x[i - p2[0].start_frame])

        if abs(prev - curr) < contact_thres and curr < contact_range_thres:
            end = i + 1
            break
        prev = curr

    return start, end

def has_contact(point, wrist_set, prev):
    contact_thres = 2.5
    contact_range_thres = 25
    y_diff_thres = 8.0
    p1 = wrist_set[0].get_subsig_at_range(point, point)
    p2 = wrist_set[1].get_subsig_at_range(point, point)

    if len(p1) > 0 and len(p2) > 0:
        y_diff = abs(p1[0].y[point - p1[0].start_frame] - p2[0].y[point - p2[0].start_frame])
        curr = abs(p1[0].x[point - p1[0].start_frame] - p2[0].x[point - p2[0].start_frame])
        return abs(prev - curr) < contact_thres and curr < contact_range_thres and y_diff < y_diff_thres, curr, True
    return False, prev, True

def find_all_peaks(subsig, valley_dir = -1):
    peaks = []
    s = subsig.speed_smooth
    pos = 0
    while True:
        pos, curr_speed = find_next_peak(s, pos + 1)
        if pos == -1:
            break
        valley, speed = find_next_valley(s, pos, curr_speed, valley_dir)
        peaks.append([pos, curr_speed, valley, subsig.start_frame + pos, s])

    return peaks



def gradual_limits_on_peaks(peaks, min_thres = 0.0):
    result = peaks.copy()
    thres = min_thres
    ending_length = starting_length = len(result)
    while starting_length * 0.5 <= ending_length and len(result) > 0:
        l = len(result) + 1
        while len(result) < l:
            thres += 0.5
            l = len(result)
            result = list(filter(lambda x: x[1] > thres, result))
        thres += 1.0
        ending_length = l
    result = list(filter(lambda x: x[0] > 10, result))
    return result, thres


def is_spike(peak):
    thres = 8.0
    s = peak[4]
    pos = peak[0]
    if 0 <= peak[0] <  len(s):
        if abs(s[pos - 1]  - s[pos])  < thres or abs(s[pos + 1]  - s[pos]) < thres:
            return False
    return True


def find_next_peak(s, pos):
    if len(s) < pos + 5:
        return -1, 0.0
    curr_speed = s[pos]
    if curr_speed >= s[pos + 1]:
        pos, curr_speed = find_next_valley(s, pos, curr_speed, 1)
        if pos == -1:
            return -1, 0.0
    while pos + 1 < len(s) and curr_speed <= s[pos + 1]:
        pos += 1
        curr_speed = s[pos]
    if pos == len(s) - 1:
        return -1, 0.0
    return pos, curr_speed


def find_prev_peak(s, pos, dir):
    if len(s) < pos + 5 * dir:
        return -1, 0.0
    curr_speed = s[pos]
    if curr_speed >= s[pos + 1 * dir]:
        pos, curr_speed = find_next_valley(s, pos, curr_speed, 1 * dir)
        if pos == -1:
            return -1, 0.0
    while pos + 1 < len(s) and curr_speed <= s[pos + 1 * dir]:
        pos += 1 * dir
        curr_speed = s[pos]
    if pos == len(s) - 1:
        return -1, 0.0
    return pos, curr_speed


def find_next_valley(s, pos, curr_speed, dir):
    while pos + 1 * dir < len(s) and pos - 1 * dir >= 0 and curr_speed > s[pos + 1 * dir]:
        pos += 1 * dir
        curr_speed = s[pos]
    if (pos + 1 >= len(s) and pos - 1 < 0):
        return -1, 0.0

    return pos, curr_speed

def find_occlusion(subsigsr, subsigsl, end):
    next_valley = 0
    prev_valley = 0
    subsigs = []
    subsigs.append(subsigsr)
    subsigs.append(subsigsl)
    occ_start = occ_end = 0
    l = len(subsigsr)
    peaksr = find_all_peaks(subsigsr[0])
    peaksl = find_all_peaks(subsigsl[0])
    peaksr, peaksl, rn, peakr, peakl = clear_noise(peaksr, peaksl)
    if subsigsr[0].end_frame <= end and len(peaksr) > 1:
        occ_start = subsigsr[0].end_frame
        prev_valley = peaksr[len(peaksr) - 1][2]

        prev_valley += subsigsr[0].start_frame - 1

        occ_end = subsigsr[l - 1].start_frame
        peaks = find_all_peaks(subsigsr[l - 1])
        if len(peaks) > 0:
            next_valley, curr_speed = find_next_valley(subsigsr[l - 1].speed_smooth, peaks[0][0], peaks[0][1], 1)
            next_valley += subsigsr[l - 1].start_frame - 1
        else:
            next_valley = subsigsr[l - 1].end_frame - 1

    l = len(subsigsl)
    if subsigsl[0].end_frame <= end and len(peaksl) > 1:
        if occ_start > subsigsl[0].end_frame:
            occ_start = subsigsl[0].end_frame
        temp = peaksl[len(peaksl) - 1][2]
        temp += subsigsl[0].start_frame - 1
        if not rn and temp < prev_valley:
            prev_valley = temp
        if occ_end < subsigsl[l - 1].start_frame:
            occ_end = subsigsl[l - 1].start_frame
        peaks = find_all_peaks(subsigsl[l - 1])
        if len(peaks) > 0:
            temp, curr_speed = find_next_valley(subsigsl[l - 1].speed_smooth, peaks[0][0], peaks[0][1], 1)
            if temp > next_valley:
                next_valley = temp
    if peakr != None and peakl != None:
        if rn and peakr[1] > peakl[1]:
            prev_valley = peakr[2] + subsigsr[0].start_frame - 1
        elif rn and peakr[1] < peakl[1]:
            prev_valley = peakl[2] + subsigsl[0].start_frame - 1
    return occ_start, occ_end, next_valley, prev_valley


def clear_noise(peaksr, peaksl):
    lr = len(peaksr)
    ll = len(peaksl)
    if lr < 3 or ll < 3:
        return peaksr, peaksl, False, None, None
    if peaksr[lr - 1][1] < peaksr[lr - 2][1] < peaksr[lr - 3][1] \
            or peaksl[ll - 1][1] < peaksl[ll - 2][1] < peaksl[ll - 3][1]:
        return peaksr, peaksl, False, None, None
    peaksr, thres = gradual_limits_on_peaks(peaksr)
    peaksl, thres = gradual_limits_on_peaks(peaksl)

    peakr = find_high(peaksr)
    peakl = find_high(peaksl)
    return peaksr, peaksl, True, peakr, peakl


def find_high(peaks):
    done = False
    if len(peaks) == 0:
        return None

    while not done:
        if (peaks[len(peaks) - 1][1] < peaks[len(peaks) - 2][1]):
            del peaks[len(peaks) - 1]
        else:
            done = True
    return peaks[len(peaks) - 1]


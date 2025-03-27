from trackedperson import TrackedPerson
from videoloader import *
import sys
import numpy
import math
from autorevise_util import joints_moving_together, connected_joint_close, gradual_limits_on_peaks, find_all_peaks, find_next_valley, is_spike, find_occlusion
from joint import JointCode
import pickle
import copy
import string

stand_threshold = 0.25


class Zone:
    def __init__(self, start, end, index):
        self.features = []
        self.start = start
        self.end = end
        self.index = index
        self.filtered_features = []

    def is_in_zone(self, start, end):
        if end <= self.end and end >= self.start:
            return True
        if start <= self.end and start >= self.start:
            return True
        if end > self.end and start < self.start:
            return True
        return False



class Activity:

    def __init__(self, start, end, label, vid_length):
        EXPAND_VAL = 0.15

        self.features = []
        self.zones = []
        self.path = ""
        self.origstart = start
        self.origend = end
        self.newstart = -1
        self.newend = sys.maxsize

        act_len = end - start
        expand = int(act_len * EXPAND_VAL) + 1
        self.start = max(start - expand, 1)
        self.end = min(end + expand, vid_length - 1)
        self.label = label

    def people_in_region(self, people):
        res = []
        for person in people:
            for i in range(self.start, self.end + 1):
                if person.is_person_here(i):
                    res.append(person)
                    break
        return res

    def split_zones(self, num_zones):
        max_features_zones = 3
        l = int((self.end - self.start)/num_zones)
        self.zones = []
        curr = self.start
        for i in range(1, num_zones + 1):
            new_end = curr + l
            self.zones.append(Zone(curr, new_end, i - 1))
            curr = new_end + 1
        self.zones[4].end = self.end

        for f in self.features:
            zone_s = list(filter(lambda x: (x.start <= f.start <= x.end), self.zones))[0]
            e_list = list(filter(lambda x: (x.start <= f.end <= x.end), self.zones))
            zone_e = e_list[0]
            if zone_s == zone_e:
                zone_s.features.append(f)
            else:
                lzs = zone_s.end - f.start
                lze = f.end - zone_e.start
                if lzs >= lze:
                    zone_s.features.append(f)
                else:
                    zone_e.features.append(f)

        for z in self.zones:
            if len(z.features) > max_features_zones:
                z.features = sorted(z.features, key=lambda feature: feature.end - feature.start)[-max_features_zones:]



class FeatureCode(enum.IntEnum):
    SIMILAR_SPEED = 0
    CONTACT = 1
    SPIKE = 2
    PEAK = 3
    OCCLUSION = 4
    FAST_RISING = 5


class Feature:
    def __init__(self, code, start, end, persons):
        self.code = code
        self.start = start
        self.end = end
        self.joint_codes = []
        self.person_letters = []
        self.persons = persons

        #option
        self.speed = -1
        self.slope = -1



class AutoreviseTemplate:

    def __init__(self):
        self.move_threshold = 1
        self.sortof_move_threshold = 0.5
        self.is_moving_threshold = 1.0
        self.sortof_move_threshold = 0.5
        self.frame_thres = 5
        self.peak_thres = 10
        self.max_peaks_person = 5
        self.length_moving_together = 5
        self.threshold_oneperson = 100
        self.threshold_minframes = 10

    def autorevise_features(self, features, activities, label, f):
        print("Revise activities\n\n")
        for activity in activities:
            maxend = newend = -1
            minstart = newstart = sys.maxsize
            print("activity...........................")
            print("Orig Start: " + str(activity.origstart) + "-- Orig end: " + str(activity.origend))

            for feature in activity.features:
                print(str(feature.code) + '-'  + str(feature.joint_codes[0]) + ' - start: ' + str(feature.start) + ' - end: ' + str(feature.end))

            #inital revisiion
            for feature in features:
                s, e = self.revise_to_first_last(feature, activity)
                minstart = min(s, minstart)
                maxend = max(e, maxend)

            people_features = self.choose_people(features, activity)
            people_features = self.find_new_endpoints(features, people_features, activity)
            self.final_pick(activity, people_features)
            print("New Start: " + str(activity.newstart) + "-- New end: " + str(activity.newend) + " Frames: "  + str(activity.newend - activity.newstart))

            f.write(activity.path + ", " + str(label) + ", " + str(activity.origstart) + ", " + str(activity.origend) + ", " + str(activity.newstart) + ", " + str(activity.newend) +"\n")
        return

    def final_pick(self, activity, people_features):
        people_features = list(filter(lambda x: x[2][1] != -1 and x[2][0] != sys.maxsize, people_features))
        if len(people_features) == 0:
            activity.newstart = activity.origstart
            activity.newend = activity.origend
            return

        for pf in people_features:
            pf[2].append((pf[2][1] - pf[2][0])/(activity.origend - activity.origstart))

        people_features = sorted(people_features, key=lambda x: x[2][2], reverse=True)
        max_points = people_features[0][2][2]
        people_features = list(filter(lambda x: x[2][2] == max_points ,people_features))
        people_features = sorted(people_features, key=lambda x: x[2][3], reverse=True)
        activity.newstart = people_features[0][2][0]
        activity.newend = people_features[0][2][1]

        if activity.newend - activity.newstart <= self.threshold_minframes:
            activity.newstart = activity.origstart
            activity.newend = activity.origend
            return

        return

    def find_new_endpoints(self, features, people_features, activity):
         for person_set in people_features:
            prior_feature = None
            newstart = sys.maxsize
            newend = -1
            points = 0
            for feature in features:
                hc = self.get_hash_code(feature[0], feature[1])
                if hc in person_set[1].keys():
                    list_features = person_set[1][hc]
                    if prior_feature is None:
                        z = activity.zones[1]
                        lf = list(filter(lambda x: x in z.features, list_features))
                        if len(lf) > 0:
                            prior_feature = sorted(lf, key=lambda x: x.start, reverse=False)[0]
                            points += 2
                        else:
                            prior_feature = sorted(list_features, key=lambda x: x.start, reverse=False)[0]
                            points += 1
                        newstart = prior_feature.start
                        newend = prior_feature.end
                    else:
                        lf = list(filter(lambda x: x.end >= newstart, list_features))
                        if len(lf) == 0:
                            lf = list_features
                        else:
                            points += 1
                        prior_feature = sorted(lf, key=lambda x: x.end, reverse=True)[0]
                        points += 1
                        newstart = min(prior_feature.start, newstart)
                        newend = max(prior_feature.end, newend)
            person_set.append([newstart, newend, points])
         return people_features



    def revise_to_first_last(self, feature, activity):
        start = activity.start
        end = activity.end
        feature_list = list(filter(lambda f: f.code == feature[0] and f.joint_codes[0] in feature[1], activity.features))
        if len(feature_list) > 0:
            start = feature_list[0].start
            end = feature_list[len(feature_list)-1].end

        return start, end

    def choose_people(self, features, activity):

        max_people = 0
        potential_features = []
        for feature in features:
            potential_features.extend(self.get_feature_matches(feature, activity.features))
            max_people = max(len(list(filter(lambda j: j != -1, feature[1]))), max_people)

        people_list = self.list_people(potential_features)

        people_features = []
        if max_people == 1 or len(people_list) == 1:
            #choose single person with most features
            for person in people_list:
                pl = list(filter(lambda f: person in f.persons, potential_features))
                matches = {}
                for feature in features:
                    lf = self.get_feature_matches(feature, pl)
                    if len(lf) > 0:
                        matches[self.get_hash_code(feature[0], feature[1])] = lf
                people_features.append([[person], matches])
            if max_people == 2: # but only one person has features
                return people_features

        for i in range(0, len(people_list)):
            for j in range(i+1, len(people_list)):
                pair = [people_list[i], people_list[j]]
                matches = {}
                for feature in features:
                    lf = self.get_feature_matches(feature, potential_features)
                    lf = self.get_people_feature_matches(feature, lf, pair)
                    if len(lf) > 0:
                        matches[self.get_hash_code(feature[0], feature[1])] = lf
                people_features.append([pair, matches])

        if len(people_features) == 0:
            return people_features
        list_pairs = sorted(people_features, key=lambda x: len(x[1].keys()), reverse=True)
        m = len(list_pairs[0][1].keys())
        list_pairs = list(filter(lambda p: len(p[1].keys()) == m, list_pairs))
        return list_pairs


    def is_in_people_features(self, feature, people_features):
        feature_list = self.get_feature_matches(feature, people_features)
        return len(feature_list) > 0


    def get_feature_matches(self, feature, features):
        return list(filter(lambda f: f.code == feature[0] and f.joint_codes[0] in feature[1] and (len(f.joint_codes) == 1 or f.joint_codes[1] in feature[1] or feature[1][1] == -1 ), features))


    def filter_features(self, feature, features):
        return list(filter(lambda f: f.code == feature.code and f.joint_codes[0] in feature.joint_codes and (
                len(f.joint_codes) == 1 or f.joint_codes[1] in feature.joint_codes or feature.joint_codes[1] == -1),
                    features))

    def get_people_feature_matches(self, feature, features, people):
        list_features = self.get_feature_matches(feature, features)
        to_remove = []
        for feat in list_features:
            for person in feat.persons:
                if not person in people:
                    to_remove.append(feat)
                    break
        for feat in to_remove:
            list_features.remove(feat)
        return list_features


    def choose_feature_match(self, prior_feature, feature_matches):
        if prior_feature != None:
            res = list(filter(lambda x: x.start > prior_feature.end, feature_matches))
            if len(res) > 0:
                return res[0]
            res = list(filter(lambda x: x.start > prior_feature.start, feature_matches))
            if len(res) > 0:
                return res[0]
        res = sorted(feature_matches, key=lambda x: x.start)
        if len(res) > 0:
            return res[0]
        else:
            return None

    def get_people_features(self, activity, pair):
        people_features = []
        for feature in activity.features:
            list_persons = list(filter(lambda x: x != -1 and (pair[0] == x or pair[1] == x), feature.persons))
            if len(list_persons) > 0:
                people_features.append(feature)
        return people_features

    def list_people(self, features):
        people_list = []
        for feature in features:
            for person in feature.persons:
                if not person in people_list:
                    people_list.append(person)
        return people_list

    def run_ut_list(self):
        # get files in UT directory
        file = ''

        here = os.path.dirname(os.path.abspath(__file__))

      #  with open(os.path.join(here, 'saved_activities_fa.pkl'), 'rb') as f:
      #     print('loading')
     #      activitytable = pickle.load(f)

    #       with open(os.path.join(here, 'fa_results.csv'), 'w') as f:

   #            for label in activitytable.keys():
  #                 res = self.select_features(activitytable[label])
 #                  #self.autorevise_features(res, activitytable[label], label, f)

#        return
        path_sigs = "./UT-interaction/"
        path = "./UT-interaction/"
        dir_list = os.listdir(path_sigs)
        activitytable = {Label.HUGGING: [], Label.HAND_SHAKING: [], Label.PUSHING: [], Label.PUNCHING: [],
                         Label.POINTING: [], Label.KICKING: []}

        used = ignored = 0
        count = 0
        total = len(dir_list)
        for label in activitytable.keys():
            activitytable[label].clear()
        all = [0 , 0]
        kept = [0 , 0]
        for file in dir_list:
            count += 1
            if ".mp4.csv" in file:
                file = file.replace(".mp4.csv", ".mp4")
            elif ".webm.csv" in file:
                file = file.replace(".webm.csv", ".mp4")
            else:
                continue
            print(file + " -- " + str(count) + " of " + str(total))
            activities, dropped, kept_all, kept, all = self.autorevise(path + file, path_sigs + file, kept, all)
            ignored += dropped
            used += kept_all
            print("Ignored so far.. " + str(ignored) + "  Used so far... " + str(used))
            for activity in activities:
                activitytable[activity.label].append(activity)
                #print(file + " " + str(activity.label) + ' ' + str(len(activity.features)) + " features ")
        print("Writing Features")
        print('saving features - 1')
        with open(os.path.join(here, 'saved_activities_fa.pkl'), 'wb') as f:
            print('saving features - 1')
            pickle.dump(activitytable, f)

        for label in activitytable.keys():
            res = self.select_features(activitytable[label])
            self.autorevise_features(res, activitytable[label], label)
        # sort activities by activity
        print("Done")

      #  return

    def get_hash_code(self, f_code, j_codes):

        return 17 + f_code * 29 + j_codes[0] * 43 + j_codes[1] * 71

    def is_person(self, list1, list2):
        if len(list2) == 0:
            return True


        for p in list1:
            if p in list2:
                return True
        return False

    def is_selected_feature(self, feat, selected_features):
        l = list(filter(lambda sf: sf[0] == feat.code and sf[1] in feat.joint_codes and (sf[2] == -1 or sf[2] in feat.joint_codes), selected_features))
        return len(l) >= 1

    def persons_in_activity(self, person_a, person_b, feature):
        for p in feature.person_letters:
            if p == person_a or p == person_b:
                continue
            else:
                return False
        return True

    def get_selected_feature(self, selected_features, feature):
        for sf in selected_features:
            if feature.code == sf[0] and sf[1] in feature.joint_codes and (sf[2] == -1 or sf[2] in feature.joint_codes):
                return sf
        return None


    def handle_in_list(self, selected_features, feature):
        sf = self.get_selected_feature(selected_features, feature)
        if sf != None:
            selected_features.remove(sf)
            val = 1
        else:
            val = 0
        return val, sf

    def add_persons_to_ordered_features(self, activities, ordered_features):
        #Review after everything else is working
        #select most freq person feature assignments
            # set first activity with person a person b
            # check each subsequent activity to try to get a match
            # if no match, invert and try again
        pass

    def order_hash_code(self, f_code, j_codes, index):
        return self.get_hash_code(f_code, j_codes) + (index+1) * 83

    def get_code_list(self, of, codes_meaning):
        of_hc = []
        for feat in of:
            f = feat[1]
            hc = self.get_hash_code(f[0], f[1:3])
            codes_meaning[hc] = [f[0], f[1:3]]
            of_hc.append(hc)
        str_hc = ''
        for hc in of_hc:
            str_hc = str_hc + str(hc) + ','
        return str_hc

    def get_feature_groups(self, people_list):
        #select unique groups
        #count how many times each group appears
        #choose two most common that encompass all features

        #not needed, nothing to evaluate
        pass

    def select_most_freq_order(self, matches):
        order_codes = {}
        codes_meaning = {}
        for m in matches:
            str_hc = self.get_code_list(m[2], codes_meaning)
            if str_hc in order_codes.keys():
                order_codes[str_hc][0] += 1
                order_codes[str_hc][1].append(m[2])
            else:
                order_codes[str_hc] = [1, [m[2]]]
        str_hc = max(order_codes, key=order_codes.get)
        values = order_codes[str_hc][1]
        of_hc = str_hc.split(',')
        order_list = []
        for hc in of_hc:
            if len(hc) > 0:
                order_list.append(codes_meaning[int(hc)])

        people_list = []
        person_two_used = False
        for val in values:
            one_letter = 'z'
            two_letter = 'z'
            one = []
            two = []
            i = -1
            for feature in val:
                i += 1
                if len(feature[2]) == 2:
                    continue
                if one_letter == 'z':
                    one_letter = feature[2][0]
                    one.append(order_list[i])
                    continue
                if one_letter == feature[2][0]:
                    one.append(order_list[i])
                    continue
                if two_letter == 'z':
                    two_letter = feature[2][0]
                    two.append(order_list[i])
                    person_two_used = True
                    continue
                if two_letter == feature[2][0]:
                    two.append(order_list[i])
                    person_two_used = True
                    continue
                print('Error')

            if len(one) > 0:
                people_list.append(one)
            if len(two) > 0:
                people_list.append(two)



        for feat in order_list:
            if self.is_in_list(feat, one):
                feat.append('person one')
            else:
                feat.append('person two')

        return order_list
    def is_in_list(self, feat, letter):
        return len(list(filter(lambda f: f[0] == feat[0] and f[1][0] == feat[1][0], letter))) > 0

    def select_best_match(self, activity_features, selected_features, people_activity):
        if len(people_activity) <= 1:
            return None
        match_list = []
        ordered_features = []
        for person_a in range(0, len(people_activity)):
            for person_b in range(person_a + 1, len(people_activity)):
                if person_b < len(people_activity) and person_a < len(people_activity):
                    selected_features_cp = copy.deepcopy(selected_features)
                    total = len(selected_features_cp)
                    count = 0
                    for f in activity_features:
                        if self.persons_in_activity(people_activity[person_a], people_activity[person_b], f):
                            val, sf = self.handle_in_list(selected_features_cp, f)
                            count += val
                    match = count/total
                    if len(people_activity) > 1:
                        match_list.append([match, people_activity[person_a], people_activity[person_b]])
        match_list = sorted(match_list, key=lambda x: x[0], reverse=True)

        if len(match_list) > 0:
            count = 0
            selected_features_cp = copy.deepcopy(selected_features)
            for f in activity_features:
                if self.persons_in_activity(match_list[0][1], match_list[0][2], f):
                    val, sf = self.handle_in_list(selected_features_cp, f)
                    if val != 0:
                        ordered_features.append([count, sf, f.person_letters])
                        count += 1
            return match_list[0][0], match_list[0], ordered_features
        return None

    def determine_feature_order(self, selected_features, activities):
        best_matches = []
        for a in activities:
            if len(a.features) == 0:
                continue
            a.selected_features = list(filter(lambda x: self.is_selected_feature(x, selected_features), a.features))
            person_letters = {}
            next_person = list(string.ascii_lowercase)
            next_person.extend(list(string.ascii_uppercase))
            for char in list(string.ascii_lowercase):
                next_person.append(str(char) + str(char))

            people_activity = []
            for f in a.selected_features:
                for p in f.persons:
                    if p in person_letters.keys():
                        f.person_letters.append(person_letters[p])
                    else:
                        if len(next_person) == 0:
                            break
                        person_letters[p] = str(next_person[0])
                        next_person.remove(next_person[0])
                        f.person_letters.append(person_letters[p])
                        people_activity.append(person_letters[p])

            match = self.select_best_match(a.selected_features, selected_features, people_activity)
            if match != None:
                best_matches.append(match)

        best_matches = sorted(best_matches, key=lambda x: x[0], reverse=True)
        best_matches_per = best_matches[int(len(best_matches)/2)][0]
        best_matches = list(filter(lambda x: x[0] == best_matches_per, best_matches))


        feature_order = self.select_most_freq_order(best_matches)
        feature_order_person = self.add_persons_to_ordered_features(activities, feature_order)
        print('Final Order')
        for f in feature_order:
            print(str(f[0]) + ' - ' + str(f[1][0]) + ' - ' + str(f[1][1]) + ' - ' + str(f[2]))
        print('--------------------')

        #print feature order
        return feature_order


    def select_features(self, activities):
        act = len(activities)
        #select common features for each actions
        features = {}
        activity_features = {}
        person_combos = []
        #joint/feature

        fcs = [FeatureCode.SPIKE, FeatureCode.PEAK, FeatureCode.OCCLUSION, FeatureCode.CONTACT, FeatureCode.SIMILAR_SPEED]
        jcs = [JointCode.WRIST, JointCode.ANKLE, JointCode.HEAD]

        for fc in fcs:
            for jc in jcs:
                for jc2 in jcs:
                    features[self.get_hash_code(fc, [jc, jc2])] = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0}
                    activity_features[self.get_hash_code(fc, [jc, jc2])] = 0
                features[self.get_hash_code(fc, [jc, -1])] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11:0, 12:0, 13:0, 14:0, 15:0}
                activity_features[self.get_hash_code(fc, [jc, -1])] = 0
        #identify important features i.e. 4 wrist spikes, 2 ankle spikes
        #identify order they occur
        #identify person id splits

        for a in activities:
            people = {}
            for z in a.zones:
                for f in z.features:
                    for person in f.persons:
                        p = person
                        if p in people.keys():
                            people[p] += 1
                        else:
                            people[p] = 1

            peops = list(people.items())
            if len(peops) > 0:
                p1 = max(peops, key=lambda x: x[1])
                peops.remove(p1)
                if len(peops) > 0:
                    p2 = max(peops, key=lambda x: x[1])
                    peops = [p1[0], p2[0]]
                else:
                    #print('No People')
                    peops = []
            else:
                peops = []
                #print('No People')
            for z in a.zones[1:4]:
                z.filtered_features = list(filter(lambda x: self.is_person(x.persons, peops), z.features))
                for f in z.filtered_features:
                    if len(f.joint_codes) > 1:
                        hc = self.get_hash_code(f.code, [f.joint_codes[0], f.joint_codes[1]])
                        activity_features[hc] += 1
                    else:
                        hc = self.get_hash_code(f.code, [f.joint_codes[0], -1])
                        activity_features[hc] += 1
            for fc in fcs:
                for jc in jcs:
                    for jc2 in jcs:
                        hc = self.get_hash_code(fc, [jc, jc2])
                        if activity_features[hc] > 0:
                            features[hc][activity_features[hc]] += 1
                        activity_features[hc] = 0
                    hc = self.get_hash_code(fc, [jc, -1])
                    if activity_features[hc] > 0:
                        features[hc][activity_features[hc]] += 1
                    activity_features[hc] = 0

        res = []
        for fc in fcs:
            for jc in jcs:
                for jc2 in jcs:
                    hc = self.get_hash_code(fc, [jc, jc2])
                    li = features[hc]
                    count = 0
                    for val in li.values():
                        count += val
                    if (count / act > 0.25):
                        res.append([fc, jc, jc2, features[hc], count])

                hc = self.get_hash_code(fc, [jc, -1])
                li = features[hc]
                count = 0
                for val in li.values():
                    count += val
                if (count / act > 0.25):
                    res.append([fc, jc, -1, features[hc], count])

        res = sorted(res, key=lambda x: x[4], reverse=True)
        res = res[:5]
        feature_order = self.determine_feature_order(res, activities)

        print("Selecting features - " + str(act) + " -- " + str(activities[0].label))
        for ele in res:
            if ele[2] != -1:
                print(str(ele[0]) + " - " + str(ele[1]) + " - " + str(ele[2]))
            else:
                print(str(ele[0]) + " - " + str(ele[1]))
        return feature_order

    def adjust_thresholds(self):
        self.is_moving_threshold += 1.0

    def reset_thresholds(self):
        self.is_moving_threshold = 1.0

    def autorevise(self, video_path, sig_path, kept, total):
        vl = VideoLoader(video_path, sig_path, "cpu")

        activities = self.get_activity_search_areas(vl.video)
        final_activities = []

        print("For " + video_path +  " Activities: " + str(len(activities)))

        for activity in activities:
            feature_list = []
            people = vl.video.tracked_persons.values()
            activity.path = video_path
            if activity.label == Label.HUGGING:
                total[0] += 1
            else:
                total[1] += 1

            people_act = activity.people_in_region(people)
            for person in people_act:
                feature_list.extend(self.get_individual(person, activity.start, activity.end))
            combo_list = []
            if len(people_act) > 1:
                for i in range(0, len(people_act)-1):
                    for j in range(i+1, len(people_act)):
                        combo_list.extend(self.get_combo(people_act[i], people_act[j], activity.start, activity.end, vl.video.length))
            else:
                continue
            if activity.label == Label.HUGGING:
                kept[0] += 1
            else:
                kept[1] += 1
            continue
            combo_list = list(filter(lambda x: (x.end - x.start >= self.frame_thres), combo_list))
            feature_list.extend(combo_list)

            feature_list = list(filter(lambda x: (x.start >= activity.start and x.end <= activity.end), feature_list))
            feature_list = sorted(feature_list, key=lambda feature: feature.start)

            activity.features = feature_list
            activity.split_zones(5)
            #print("Split zones: " + str(len(activity.zones)))
            final_activities.append(activity)
        dropped = len(activities) - len(final_activities)
        return final_activities, dropped, len(final_activities), kept, total

    def get_activity_search_areas(self, video):
        res = []
        for lr in video.label_entries:
            res.append(Activity(lr.start, lr.end, lr.label, video.length))

        return res

    def get_individual(self, person_a, start, end):
        res = []
        for joint in person_a.joints:
            list_features = self.find_peaks(joint, person_a.id)
            list_features = list(filter(lambda x: (x.start >= start and x.end <= end), list_features))

            for f in list_features:
                f.joint_codes.append(joint.code)
            res.extend(list_features)

        res.extend(self.find_wrist_occlusions(person_a, start, end))
        res = sorted(res, key=lambda feature: feature.speed)
        if len(res) > self.max_peaks_person:
            res = res[-self.max_peaks_person:]
        return res


    def remove_matching(self, list):
        done = False
        index = 0
        if len(list) < 2:
            return list
        while not done:
            check = list[index]
            i = index + 1
            done2 = i >= len(list)
            while not done2:
                if check.end == list[i].end and check.start == list[i].start and check.joint_codes[0] == list[i].joint_codes[0] and check.joint_codes[1] == list[i].joint_codes[1]:
                    toremove = list[i]
                    list.remove(toremove)
                else:
                    i += 1
                done2 = i >= len(list)
            index += 1
            done = index >= len(list)
        return list

    def get_combo(self, person_a, person_b, start, end, vid_length):
        res_len = sys.maxsize
        while res_len > self.max_peaks_person:
            res = []
            occ = []
            for joint_a in person_a.joints:
                for joint_b in person_b.joints:
                    list_features = self.find_same_speed_and_contact(joint_a, joint_b, start, end, [person_a.id, person_b.id])
                    for f in list_features:
                        f.joint_codes.append(joint_a.code)
                        f.joint_codes.append(joint_b.code)
                    res.extend(list_features)
            res = self.remove_matching(res)
            occ.extend(self.find_incomplete_occlusions(person_a, person_b, start, end, vid_length))
            if len(occ) > 0:
                res.append(self.choose_max_occ(occ))
            self.adjust_thresholds()
            res_len = len(res)
        self.reset_thresholds()
        return res


    def find_same_speed_and_contact(self, joint_a, joint_b, start, end, person_ids):
        res = []
        joint_set = [joint_a, joint_b]
        segs = joints_moving_together(joint_set, start, end, self.is_moving_threshold)
        for s in segs:
            s[0] += start
            s[1] += start
        segs = list(filter(lambda x: (x[1] > start and x[0] < end), segs))
        segs = sorted(segs, key=lambda x: x[1] - x[0], reverse=True)
        if len(segs) > 0:
            seg = segs[0]
            if joint_a.code == joint_b.code:
                f = Feature(FeatureCode.SIMILAR_SPEED, seg[0], seg[1], person_ids)
                res.append(f)
            #CHECK: Value returned when no contact found
            start, end = connected_joint_close(seg, joint_set)
            if end - start > 0:
                f = Feature(FeatureCode.CONTACT, start, end, person_ids)
                res.append(f)
        res = list(filter(lambda x: x.code == FeatureCode.CONTACT or (x.end - x.start) > self.length_moving_together, res))
        return res

    def is_fast_rising(self, p, spd):
        thres = .6
        if spd/p[1] > thres:
            return True
        return False

    def find_peaks(self, joint_a, person_id):
        res = []
        peaks = []
        for s in joint_a.subsigs:
            peaks, thres = gradual_limits_on_peaks(find_all_peaks(s))
            for p in peaks:
                v, spd = find_next_valley(p[4], p[0], p[1], 1)
                if is_spike(p):
                    f = Feature(FeatureCode.SPIKE, p[2] + s.start_frame, v + s.start_frame, [person_id])
                    f.speed = p[1]
                    res.append(f)
                #else:
                    #f = Feature(FeatureCode.PEAK, p[2]+ s.start_frame, v + s.start_frame, [person_id])
                    #f.speed = p[1]
                    #res.append(f)
        return res

    def find_wrist_occlusions(self, person_a, start, end):
        res = []

        if len(person_a.left_wrist.subsigs) == 0 or len(person_a.right_wrist.subsigs) == 0:
            return res
        if person_a.start_frame > start or person_a.end_frame < end:
            return res

        occ_start, occ_end, next_valley, prev_valley = find_occlusion(person_a.right_wrist.subsigs, person_a.left_wrist.subsigs, end)
        if occ_start > 0 and occ_end > 0:

            if occ_start > occ_end:
                feature = Feature(FeatureCode.OCCLUSION, occ_end, occ_start, [person_a.id])
            else:
                feature = Feature(FeatureCode.OCCLUSION, occ_start, occ_end, [person_a.id])
            feature.joint_codes = [JointCode.WRIST, -1]
            feature.distance = abs(occ_end - occ_start)
            res.append(feature)
            return res
        return res

    def get_dist(self, joint_a, joint_b):
        s = joint_a.subsigs[len(joint_a.subsigs) - 1]
        x1 = s.x[len(s.x) - 1]
        y1 = s.y[len(s.x) - 1]
        s = joint_b.subsigs[0]
        x2 = s.x[0]
        y2 = s.y[0]
        return math.sqrt((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1))

    def find_incomplete_occlusions(self, person_a, person_b, start, end, length):
        res = []
        if person_a.end_frame < person_b.start_frame:
            if person_a.end_frame > start and person_b.start_frame < end:
                feature = Feature(FeatureCode.OCCLUSION, person_a.end_frame, person_b.start_frame, [person_a.id])
                feature.joint_codes = [JointCode.WRIST, -1]
                if len(person_a.right_wrist.subsigs) > 0 and len(person_b.right_wrist.subsigs) > 0 and len(person_a.left_wrist.subsigs) > 0 and len(person_b.left_wrist.subsigs) > 0 :
                    distancer = self.get_dist(person_a.right_wrist, person_b.right_wrist)
                    distancel = self.get_dist(person_a.left_wrist, person_b.left_wrist)
                    nearest = min(distancel, distancer)
                    if nearest < self.threshold_oneperson:
                        feature.distance = abs(person_a.end_frame - person_b.start_frame)
                        res.append(feature)
        elif person_b.end_frame < person_a.start_frame:
            if person_b.end_frame > start and person_a.start_frame < end:
                feature = Feature(FeatureCode.OCCLUSION, person_b.end_frame, person_a.start_frame, [person_b.id])
                feature.joint_codes = [JointCode.WRIST, -1]
                if len(person_a.right_wrist.subsigs) > 0 and len(person_b.right_wrist.subsigs) > 0 and len(person_a.left_wrist.subsigs) > 0 and len(person_b.left_wrist.subsigs) > 0 :
                    distancer = self.get_dist(person_b.right_wrist, person_a.right_wrist)
                    distancel = self.get_dist(person_b.left_wrist, person_a.left_wrist)
                    nearest = min(distancel, distancer)
                    if nearest < self.threshold_oneperson:
                        feature.distance = abs(person_b.end_frame - person_a.start_frame)
                        res.append(feature)

        return res


    def choose_max_occ(self, occ):
        o =  sorted(occ, key=lambda feature: feature.distance)[0]
        return o

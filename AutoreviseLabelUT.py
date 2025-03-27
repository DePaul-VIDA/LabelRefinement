from videoloader import Label


class AutoreviseLabelUT:

    def __init__(self, values):
        self.label = Label.from_str(values[1])
        self.video_path = values[0]
        self.origstart = int(values[2])
        self.origend = int(values[3])
        self.newstart = int(values[4])
        self.newend = int(values[5])
        self.VL = None
        self.better = ""

    def get_start(self, revised, doingend):
        if doingend:
            if revised:
                return self.newend - 5
            else:
                return self.origend - 5
        else:
            if revised:
                return self.newstart
            else:
                return self.origstart

    def get_end(self, revised):
        if revised:
            return self.newend
        else:
            return self.origend


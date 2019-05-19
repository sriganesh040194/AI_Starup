class Threshold:
    def __init__(self, filename):
        self.filename = filename

    def get(self):
        if self.filename == "Sample01.mp4":
            return 0.7
        elif self.filename == "Demo.mp4":
            return 0.96
        elif self.filename == "Traffic.mp4":
            return 0.7
        elif self.filename == "rtsp_person.mp4":
            return 0.96
        elif self.filename == "rtsp_bird.mp4":
            return 0.99
        elif self.filename == "Traffic2.mp4":
            return 0.7
        return 0.96
        






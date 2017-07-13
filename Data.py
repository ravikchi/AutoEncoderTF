class Data:
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels
        self.index = 0
        self.size = len(input)

    def inp_size(self):
        return len(self.input[0])

    def next_batch(self, batchSize):
        cur_index = self.index
        if self.index == len(self.input):
            self.index = 0
            cur_index = self.index

        if (self.index + batchSize) > len(self.input):
            self.index = len(self.input)
        else:
            self.index += batchSize

        return self.input[cur_index:self.index], self.labels[cur_index:self.index]
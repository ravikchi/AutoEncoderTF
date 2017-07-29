from Data import Data

class Validation(Data):
    def __init__(self, input, labels, validSize):
        Data.__init__(self, input, labels)
        self.validSize = int(self.size * validSize)
        self.input = input[:-self.validSize]
        self.labels = labels[:-self.validSize]
        self.validInput = input[-self.validSize:]
        self.validLabels = labels[-self.validSize:]
        self.valIndex = 0

    def next_validation_batch(self, batchSize):
        cur_index = self.valIndex
        if self.index == len(self.validInput):
            self.valIndex = 0
            cur_index = self.valIndex

        if (self.valIndex + batchSize) > len(self.validInput):
            self.valIndex = len(self.validInput)
        else:
            self.valIndex += batchSize

        return self.validInput[cur_index:self.valIndex], self.labels[cur_index:self.valIndex]
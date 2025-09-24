

class EarlyStopper:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.bestLoss = None


    def earlyStop(self, loss):
        if self.bestLoss is None:
            self.bestLoss = loss
        elif loss >= self.bestLoss:
            self.counter += 1
            if (self.counter > self.patience):
                return True
        else:
            self.bestLoss = loss
            self.counter = 0
import math

class ActivationFunction:
    # __singleton = None
    # def __new__(cls):
    #     cls.__singleton = cls
    def __init__(self, types='Sigmoid'):
        # if self.__singleton is None:
        #     self.__singleton = self
        # cls = self.__singleton
        self.func = self.sigmoid_function
        if types == 'Sigmoid':
            self.func = self.sigmoid_function

    def run(self):
        return self.func(x)

    def sigmoid_function(self, x):
        return 1 / (1 + math.exp(-x))

if __name__ == '__main__':
    myfunc = ActivationFunction('Sigmoid')

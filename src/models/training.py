import models.differentiable as dm

# This class defines a device to train a model using gradient descent. It takes two
# parameters: a model, as defined in model.py, and a generator function, which generates
# examples on which to train.
class TrainingRig():
    def __init__(self, data_generator, learning_schedule, batch_size):
        self.dg = data_generator
        self.sched = learning_schedule


    def train(self, model: dm.DifferentiableModel):
        if iterations < 1:
            return
        
        for i in range(1, iterations):
            for n in range(0, batch_size - 1):
                gradients = model.loss_gradient(self.data_generator(n))
                self.descend(model, gradients, self.sched(i))


    def descend(self, model, direction, learning_rate):
        ps = model.get_parameters()
        new = []
        for (p, g) in zip(ps, direction):
            new.append(p - learning_rate * g)
        model.set_parameters(new)
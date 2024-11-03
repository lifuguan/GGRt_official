
class Model(object):
    def __init__(self, args) -> None:
        self.args = args

    def to_distributed(self):
        raise NotImplementedError

    def switch_to_eval(self):
        raise NotImplementedError

    def switch_to_train(self):
        raise NotImplementedError

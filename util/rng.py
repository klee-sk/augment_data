import random

_rng = random.Random()

def init(seed):
    _rng.seed(seed)

def get():
    return random.Random(_rng.random())




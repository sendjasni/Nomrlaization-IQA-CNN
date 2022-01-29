from constants import *
import main
import main_preprocess
import main_train_from_npy

env = 'LOCAL'

# main.run(env)
main_preprocess.run(env)
# main_train_from_npy.run(env)

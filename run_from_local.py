import constants as cst
import main
import main_preprocess
import main_train_from_npy

env = 'LOCAL'

main_preprocess.run(env, cst)
# main_train_from_npy.run(env, cst)




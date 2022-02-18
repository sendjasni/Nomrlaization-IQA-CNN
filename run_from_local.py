import constants as cst
import main
import main_preprocess
import main_load_preprocess_LIVE
import main_train_from_npy
import main_train_from_npy_LIVE

env = 'LOCAL'

# main.run(env, cst)

# if cst.DATABASE_NAME == "LIVE2":
#     main_load_preprocess_LIVE.run(env, cst)
# else:
main_preprocess.run(env, cst)

# if cst.DATABASE_NAME == "LIVE2":
#     main_train_from_npy_LIVE.run(env, cst)
# else:
# main_train_from_npy.run(env, cst)




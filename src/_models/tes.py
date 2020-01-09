from os import scandir
from os.path import basename
from keras.models import load_model
import pickle
import dill
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def testfun():
    DATASET_FOLDER = "/data/students_home/amoscatelli/Desktop/actionAnalysis/datasets/"
    SAVED_MODEL_FOLDER = "/data/students_home/amoscatelli/Desktop/actionAnalysis/savedModels/"
    savedModels = [f for f in scandir(SAVED_MODEL_FOLDER) if f.path[-3:] == ".h5"]
    model_names = ["PoseNet-101", "keypoint_rcnn_X_101_32x8d_FPN_3x"]
    results = []
    for model_to_analyse in model_names:
        datasetName = DATASET_FOLDER + model_to_analyse + "-SPLIT-dataset.pickle"
        with open(datasetName, 'rb') as file_in:
            train_set, val_set, test_set = pickle.load(file_in)
        for saved_model in savedModels:
            #         modelName, preprocess_functions, normalise = fromFileNameToParameters(saved_model)
            modelName, normalise = fromFileNameToModel(saved_model)
            if modelName != model_to_analyse:
                continue

            ### LOAD accessories
            accessoriesPath = saved_model.path[:-3] + ".pickle"

            with open(accessoriesPath, "rb") as handle:
                res = pickle.load(handle)

            loaded_functions = [dill.loads(x) for x in res["prep_fun_DILL"]]

            one_hot_encoding = loaded_functions[0]
            normaliseBeforePadding = loaded_functions[1]
            paddingTrainValTest = loaded_functions[2]
            preprocessData = loaded_functions[3]
            specificFunction = loaded_functions[4]

            X_train, y_train, X_val, y_val, X_test, y_test = preprocessData(train_set, val_set, test_set, normalise,
                                                                            specificFunction)
            #         ## reshaping ###
            #         X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 17 * 2)
            #         X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 17 * 2)
            #         X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 17 * 2)
            loadedModel = load_model(SAVED_MODEL_FOLDER + saved_model.name)
            print(basename(saved_model))
            val_acc, test_acc = plotValTestResult(loadedModel, X_val, y_val, X_test, y_test)
            results.append((val_acc, test_acc, saved_model, res["history"]))
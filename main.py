# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import os
import json
import math
import torch
import numpy
import pandas
import argparse
from time import gmtime, strftime

import scikit_wrappers


def get_time_str():
    return strftime("%Y%m%d_%H%M%S", gmtime())

def print_and_log(my_str, log_name):
    out = '{}|{}'.format(get_time_str(), my_str)
    print(out)
    with open(log_name, 'a') as f_log:
        print(out, file=f_log)      

def load_UCR_dataset(path, dataset):
    """
    Loads the UCR dataset given in input in numpy arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    train_file = os.path.join(path, dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join(path, dataset, dataset + "_TEST.tsv")
    train_df = pandas.read_csv(train_file, sep='\t', header=None)
    test_df = pandas.read_csv(test_file, sep='\t', header=None)
    train_array = numpy.array(train_df)
    test_array = numpy.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = numpy.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = numpy.expand_dims(train_array[:, 1:], 1).astype(numpy.float64)
    train_labels = numpy.vectorize(transform.get)(train_array[:, 0])
    test = numpy.expand_dims(test_array[:, 1:], 1).astype(numpy.float64)
    test_labels = numpy.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train, train_labels, test, test_labels
    mean = numpy.nanmean(numpy.concatenate([train, test]))
    var = numpy.nanvar(numpy.concatenate([train, test]))
    train = (train - mean) / math.sqrt(var)
    test = (test - mean) / math.sqrt(var)
    return train, train_labels, test, test_labels


def fit_hyperparameters(file, train, train_labels, cuda, gpu,
                        save_memory=False):
    """
    Creates a classifier from the given set of hyperparameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    classifier = scikit_wrappers.CausalCNNEncoderClassifier()

    # Loads a given set of hyperparameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    # Check the number of input channels
    params['in_channels'] = numpy.shape(train)[1]
    params['cuda'] = cuda
    params['gpu'] = gpu
    classifier.set_params(**params)
    return classifier.fit(
        train, train_labels, save_memory=save_memory, verbose=True
    )

def run(dataset, gpu, log_name):

    path = '/localscratch/shared/ts/UCRArchive_2018'
    save_path = '/localscratch/shared/ts/exp'
    cuda = True
    hyper = 'default_hyperparameters.json'
    load = False
    fit_classifier = False

    train, train_labels, test, test_labels = load_UCR_dataset(path, dataset)
    print(bool(numpy.isnan(numpy.sum(train))))
    return

    if not load and not fit_classifier:
        classifier = fit_hyperparameters(hyper, train, train_labels, cuda, gpu)
    else:
        classifier = scikit_wrappers.CausalCNNEncoderClassifier()
        hf = open(os.path.join(save_path, dataset + '_hyperparameters.json'), 'r')
        hp_dict = json.load(hf)
        hf.close()
        hp_dict['cuda'] = cuda
        hp_dict['gpu'] = gpu
        classifier.set_params(**hp_dict)
        classifier.load(os.path.join(save_path, dataset))

    if not load:
        if fit_classifier:
            classifier.fit_classifier(classifier.encode(train), train_labels)
        classifier.save(os.path.join(save_path, dataset))
        with open(os.path.join(save_path, dataset + '_hyperparameters.json'), 'w') as fp:
            json.dump(classifier.get_params(), fp)

    print_and_log("Test accuracy: " + str(classifier.score(test, test_labels)), log_name)
    
    
if __name__ == '__main__':

    dataset_list_1 = ['Adiac','ArrowHead','Beef','BeetleFly','BirdChicken','Car','CBF','ChlorineConcentration','CinCECGTorso','Coffee','Computers','CricketX','CricketY','CricketZ','DiatomSizeReduction','DistalPhalanxOutlineCorrect','DistalPhalanxOutlineAgeGroup','DistalPhalanxTW','Earthquakes','ECG200','ECG5000','ECGFiveDays','ElectricDevices','FaceAll','FaceFour','FacesUCR','FiftyWords','Fish','FordA','FordB','GunPoint','Ham','HandOutlines','Haptics','Herring','InlineSkate','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lightning2','Lightning7','Mallat','Meat','MedicalImages','MiddlePhalanxOutlineCorrect','MiddlePhalanxOutlineAgeGroup','MiddlePhalanxTW','MoteStrain','NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2','OliveOil','OSULeaf','PhalangesOutlinesCorrect','Phoneme','Plane']

    dataset_list_2 = ['ProximalPhalanxOutlineCorrect','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxTW','RefrigerationDevices','ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances','SonyAIBORobotSurface1','SonyAIBORobotSurface2','StarLightCurves','Strawberry','SwedishLeaf','Symbols','SyntheticControl','ToeSegmentation1','ToeSegmentation2','Trace','TwoLeadECG','TwoPatterns','UWaveGestureLibraryX','UWaveGestureLibraryY','UWaveGestureLibraryZ','UWaveGestureLibraryAll','Wafer','Wine','WordSynonyms','Worms','WormsTwoClass','Yoga','ACSF1','BME','Chinatown','Crop','EOGHorizontalSignal','EOGVerticalSignal','EthanolLevel','FreezerRegularTrain','FreezerSmallTrain','Fungi','GunPointAgeSpan','GunPointMaleVersusFemale','GunPointOldVersusYoung','HouseTwenty','InsectEPGRegularTrain','InsectEPGSmallTrain','MixedShapesRegularTrain','MixedShapesSmallTrain','PigAirwayPressure','PigArtPressure','PigCVP','PowerCons','Rock','SemgHandGenderCh2','SemgHandMovementCh2','SemgHandSubjectCh2','SmoothSubspace','UMD']

    dataset_list_varied = ['AllGestureWiimoteX','AllGestureWiimoteY','AllGestureWiimoteZ','DodgerLoopDay','DodgerLoopGame','DodgerLoopWeekend','GestureMidAirD1','GestureMidAirD2','GestureMidAirD3','GesturePebbleZ1','GesturePebbleZ2','MelbournePedestrian','PickupGestureWiimoteZ','PLAID','ShakeGestureWiimoteZ']

    for dataset in dataset_list_1:
        run(dataset=dataset, gpu=0, log_name='dataset_list_1.txt')

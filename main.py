########################################################
#Project name: Infocom
#Date: 14/July/2021
########################################################
from __future__ import division

import os
import csv
import argparse
# import h5py
import pickle
import numpy as np
from tqdm import tqdm
import random
from time import time
import torch.nn.functional as F

# from custom_metrics import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
import sklearn
from sklearn.metrics import precision_recall_fscore_support as precision_recall_fscore
import torch
from sklearn.model_selection import train_test_split


############################
# Fix the seed
############################
seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)
random.seed(seed)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False

def show_all_files_in_directory(input_path,extension):
    'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(extension):
               files_list.append(os.path.join(path, file))
    return files_list

def open_npz(path,key):
    data = np.load(path)[key]
    return data

def save_npz(path,train_name,train_data,val_name,val_data):
    check_and_create(path)
    np.savez_compressed(path+train_name, train=train_data)
    np.savez_compressed(path+val_name, val=val_data)


def context_aware_label(c1, c2, c3, c4): # UPDATED FOR TL BY DR
    label_c1 =  np.zeros((c1.shape[0], 4))
    label_c2 =  np.zeros((c2.shape[0], 4))
    label_c3 =  np.zeros((c3.shape[0], 4))
    label_c4 =  np.zeros((c4.shape[0], 4))
    label_c1[:, 0] = 1
    label_c2[:, 1] = 1
    label_c3[:, 2] = 1
    label_c4[:, 3] = 1
    return label_c1, label_c2, label_c3, label_c4

def precison_recall_F1(model,Xtest,Ytest):
    #####For recall and precison
    y_pred = model.predict(Xtest)
    y_pred_bool = np.argmax(y_pred, axis=1)
    y_true_bool = np.argmax(Ytest, axis=1)
    return precision_recall_fscore(y_true_bool, y_pred_bool,average='weighted')


def detecting_related_file_paths(path,categories,episodes):
    find_all_paths =['\\'.join(a.split('\\')[:-1]) for a in show_all_files_in_directory(path,'rf.npz')]     # rf for example
    # print('find_all_paths',find_all_paths)
    selected = []
    for Cat in categories:   # specify categories as input
        for ep in episodes:
            selected = selected + [s for s in find_all_paths if Cat in s.split('\\') and 'episode_'+str(ep) in s.split('\\')]
    print('Getting {} data out of {}'.format(len(selected),len(find_all_paths)))

    return selected

def get_data(data_paths,modality,key,test_on_all,path_test_all):   # per cat for now, need to add per epside for FL part
    first = True
    for l in tqdm(data_paths):
        randperm = np.load(l+'\\ranperm.npy')
        if first == True:
            open_file = open_npz(l+'\\'+modality+'.npz',key)
            train_data = open_file[randperm[:int(0.8*len(randperm))]]
            validation_data = open_file[randperm[int(0.8*len(randperm)):int(0.9*len(randperm))]]
            test_data = open_file[randperm[int(0.9*len(randperm)):]]
            first = False
        else:
            open_file = open_npz(l+'\\'+modality+'.npz',key)
            train_data = np.concatenate((train_data, open_file[randperm[:int(0.8*len(randperm))]]),axis = 0)
            validation_data = np.concatenate((validation_data, open_file[randperm[int(0.8*len(randperm)):int(0.9*len(randperm))]]),axis = 0)
            test_data = np.concatenate((test_data, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)


        ####PER CAT
        if 'Cat1' in l.split('\\'):
            try:
                test_data_cat1 = np.concatenate((test_data_cat1, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
            except NameError:
                test_data_cat1 = open_file[randperm[int(0.9*len(randperm)):]]

        elif 'Cat2' in l.split('\\'):
            try:
                test_data_cat2 = np.concatenate((test_data_cat2, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
            except NameError:
                test_data_cat2 = open_file[randperm[int(0.9*len(randperm)):]]

        elif 'Cat3' in l.split('\\'):
            try:
                test_data_cat3 = np.concatenate((test_data_cat3, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
            except NameError:
                test_data_cat3 = open_file[randperm[int(0.9*len(randperm)):]]

        elif 'Cat4' in l.split('\\'):
            try:
                test_data_cat4 = np.concatenate((test_data_cat4, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
            except NameError:
                test_data_cat4 = open_file[randperm[int(0.9*len(randperm)):]]

    if test_on_all:
        test_data = open_npz(path_test_all+'\\'+modality+'_'+'all.npz',key)
        test_data_cat1 = open_npz(path_test_all+'\\'+modality+'_'+'cat1.npz',key)
        test_data_cat2 = open_npz(path_test_all+'\\'+modality+'_'+'cat2.npz',key)
        test_data_cat3 = open_npz(path_test_all+'\\'+modality+'_'+'cat3.npz',key)
        test_data_cat4 = open_npz(path_test_all+'\\'+modality+'_'+'cat4.npz',key)

    print('categories shapes',test_data_cat1.shape,test_data_cat2.shape,test_data_cat3.shape,test_data_cat4.shape)
    print('tr/val/te',train_data.shape,validation_data.shape,test_data.shape)
    return train_data,validation_data,test_data, test_data_cat1, test_data_cat2, test_data_cat3, test_data_cat4


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc

 ### ADDED FOR TL BY DR #########
def top_k_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        # print("shape y_pred", y_pred.shape)
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
        # print("shape y_pred2", y_pred.shape)

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target = torch.argmax(target, dim=1) #CONVERTING TO LABELS AGAIN (ADDED BY DR)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc.cpu().detach().numpy())
        return list_topk_accs # list of topk accuracies for entire batch [topk1, topk2, ... etc]
 ### END OF ADDITION FOR TL BY DR #########


parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--id_gpu', default=1, type=int, help='which gpu to use.')
parser.add_argument('--data_folder', help='Location of the data directory', type=str, default="D:\\FL\\data_half_half_size\\")
parser.add_argument('--input', nargs='*', default=['img', 'coord', 'lidar'], choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.')

parser.add_argument('--epochs', default=150, type = int, help='Specify the epochs to train')
parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--bs',default=32, type=int,help='Batch size')
parser.add_argument('--shuffle', help='shuffle or not', type=str2bool, default =True)

parser.add_argument('--test_all', help='test on all data', type=str2bool, default =True)
parser.add_argument('--test_all_path', help='Location of all test', type=str,default = 'D:\\FL\\baseline_code\\all_test\\')


parser.add_argument('--restore_models', type=str2bool, help='Load single modality trained weights', default=False)
parser.add_argument('--model_folder', help='Location of the trained models folder', type=str,default = 'D:\\FLASH_Context_Prediction\\model_folder\\')

parser.add_argument('--experiment_catergories', nargs='*' ,default=['Cat1','Cat2','Cat3','Cat4'], help='categories included',choices=['Cat1','Cat2','Cat3','Cat4'])
parser.add_argument('--experiment_epiosdes', nargs='*' ,default=['0','1','2','3','4','5','6','7','8','9'], help='episodes included',choices=['0','1','2','3','4','5','6','7','8','9'])

#*** Fusion specific arguments **********
parser.add_argument('--fusion_layer', type=str, help='Assign the layer name where the fusion to be performed.', default='penultimate')
parser.add_argument('--retrain', type=str2bool, help='Retrain the model on loaded weights', default=True)

args = parser.parse_args()
print('Argumen parser inputs', args)

if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)
torch.manual_seed(1234)
check_and_create(args.model_folder)


print('******************Detecting related file paths*************************')
selected_paths = detecting_related_file_paths(args.data_folder,args.experiment_catergories,args.experiment_epiosdes)
###############################################################################
# Outputs ##needs to be changed
###############################################################################
# print('******************Getting RF data*************************')
# RF_train, RF_val, RF_test,RF_c1,RF_c2,RF_c3,RF_c4 = get_data(selected_paths,'rf','rf',args.test_all,args.test_all_path)
# print('RF data shapes on same client',RF_train.shape,RF_val.shape,RF_test.shape)
#


###############################################################################
# Inputs #
###############################################################################

if 'coord' in args.input:
    print('******************Getting Gps data*************************')
    X_coord_train, X_coord_validation, X_coord_test,gps_c1,gps_c2,gps_c3,gps_c4 = get_data(selected_paths,'gps','gps',args.test_all,args.test_all_path)
    print('GPS data shapes',X_coord_train.shape,X_coord_validation.shape,X_coord_test.shape)
    coord_train_input_shape = X_coord_train.shape

    ### normalized
    print('max over dataset b', np.max(abs(X_coord_train)),np.max(abs(X_coord_validation)),np.max(abs(X_coord_test)))
    X_coord_train = X_coord_train / 9747
    X_coord_validation = X_coord_validation / 9747
    X_coord_test = X_coord_test / 9747
    gps_c1 = gps_c1/ 9747
    gps_c2 = gps_c2/ 9747
    gps_c3 = gps_c3/ 9747
    gps_c4 = gps_c4/ 9747
    ## For convolutional input
    X_coord_train = X_coord_train.reshape((X_coord_train.shape[0], X_coord_train.shape[1], 1))
    X_coord_validation = X_coord_validation.reshape((X_coord_validation.shape[0], X_coord_validation.shape[1], 1))
    X_coord_test = X_coord_test.reshape((X_coord_test.shape[0], X_coord_test.shape[1], 1))
    print('shapes after re-shaping',X_coord_train.shape)

    gps_c1 = gps_c1.reshape((gps_c1.shape[0], gps_c1.shape[1], 1))
    gps_c2 = gps_c2.reshape((gps_c2.shape[0], gps_c2.shape[1], 1))
    gps_c3 = gps_c3.reshape((gps_c3.shape[0], gps_c3.shape[1], 1))
    gps_c4 = gps_c4.reshape((gps_c4.shape[0], gps_c4.shape[1], 1))

    saved_file_name = 'coord'


if 'img' in args.input:
    print('******************Getting image data*************************')
    X_img_train, X_img_validation, X_img_test,img_c1,img_c2,img_c3,img_c4 = get_data(selected_paths,'image','img',args.test_all,args.test_all_path)
    print('max over dataset b', np.max(X_img_train),np.max(X_img_validation),np.max(X_img_test))
    print('image data shapes',X_img_train.shape,X_img_validation.shape,X_img_test.shape)
    ###normalize images
    X_img_train = X_img_train / 255
    X_img_validation = X_img_validation / 255
    X_img_test = X_img_test/255
    img_c1 = img_c1/ 255
    img_c2 = img_c2/ 255
    img_c3 = img_c3/ 255
    img_c4 = img_c4/ 255
    img_train_input_shape = X_img_train.shape

    saved_file_name = 'img'

if 'lidar' in args.input:
    print('******************Getting lidar data*************************')
    X_lidar_train, X_lidar_validation, X_lidar_test,lid_c1,lid_c2,lid_c3,lid_c4 = get_data(selected_paths,'lidar','lidar',args.test_all,args.test_all_path)
    print('lidar data shapes',X_lidar_train.shape,X_lidar_validation.shape,X_lidar_test.shape)
    lidar_train_input_shape = X_lidar_train.shape

    saved_file_name = 'lidar'


# LABELING FOR CONTEXT AWARE PREDICTIONS
num_classes = len(args.experiment_catergories)
if 'coord' in args.input:
    label_c1, label_c2, label_c3, label_c4 = context_aware_label(gps_c1, gps_c2, gps_c3, gps_c4)
    X_coord = np.concatenate((gps_c1, gps_c2, gps_c3, gps_c4), axis=0)
    labels = np.concatenate((label_c1, label_c2, label_c3, label_c4), axis=0)
    print("TESTTTTT: ", X_coord.shape, X_coord.shape)
    X_coord_train,  X_coord_test, ytrain, ytest = train_test_split(X_coord, labels, test_size=0.1, shuffle = True, random_state=42)  # 90/10 is train/test size
    X_coord_train, X_coord_validation, ytrain, yval = train_test_split(X_coord_train, ytrain, test_size=0.1, shuffle=True,
                                                                  random_state=42)  # 81/9 is train/val size

if 'img' in args.input:
    label_c1, label_c2, label_c3, label_c4 = context_aware_label(img_c1,img_c2,img_c3,img_c4)
    X_img = np.concatenate((img_c1,img_c2,img_c3,img_c4), axis=0)
    labels = np.concatenate((label_c1, label_c2, label_c3, label_c4), axis=0)
    X_img_train,  X_img_test, ytrain, ytest = train_test_split(X_img, labels, test_size=0.1, shuffle = True, random_state=42)  # 90/10 is train/test size
    X_img_train, X_img_validation, ytrain, yval = train_test_split(X_img_train, ytrain, test_size=0.1, shuffle=True,
                                                                  random_state=42)  # 81/9 is train/val size


if 'lidar' in args.input:
    label_c1, label_c2, label_c3, label_c4 = context_aware_label(lid_c1,lid_c2,lid_c3,lid_c4)
    X_lidar = np.concatenate((lid_c1,lid_c2,lid_c3,lid_c4), axis=0)
    labels = np.concatenate((label_c1, label_c2, label_c3, label_c4), axis=0)
    X_lidar_train,  X_lidar_test, ytrain, ytest = train_test_split(X_lidar, labels, test_size=0.1, shuffle = True, random_state=42)  # 90/10 is train/test size
    X_lidar_train, X_lidar_validation, ytrain, yval = train_test_split(X_lidar_train, ytrain, test_size=0.1, shuffle=True,
                                                                  random_state=42)  # 81/9 is train/val size



# train, validation and  test seperation



print('******************Succesfully generated the data*************************')

##############################################################################
# Model configuration
##############################################################################

# start_time = time.time()
fusion = False
if len(args.input) >1: fusion = True
test_acc = [] #lISTING ALL THE TEST ACCURACIES

################################################################
# CONVERTING NUMPY ARRAY TO TORCH #
################################################################
# Implementing in pytorch
import torch
import torchvision


from ModelHandler_pytorch import CoordNet, CameraNet, LidarNet, InfoFusionThree, FeatureFusion



#############################################################################################################################
# WORKING ON EACH INDIVIDUAL MODALITIES
#############################################################################################################################
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

if fusion == False:

    if 'coord' in args.input:
        xtrain = X_coord_train
        xval = X_coord_validation
        xtest = X_coord_test
        print("XTRAIN COORDINATE SHAPE", X_coord_train.shape)
        #input_size = list(list(xtrain_image.values())[0][0].shape)
        model = CoordNet(input_dim=xtrain.shape[1], output_dim=num_classes)

    if 'img' in args.input:
        xtrain = X_img_train
        xval = X_img_validation
        xtest = X_img_test
        model = CameraNet(input_dim=xtrain.shape[1], output_dim=num_classes)

    if 'lidar' in args.input:
        xtrain = X_lidar_train
        xval = X_lidar_validation
        xtest = X_lidar_test
        model = LidarNet(input_dim=xtrain.shape[1], output_dim=num_classes)

    print("Shape of X and Y train data: ", xtrain.shape, ytrain.shape)
    print("Shape of X and Y val data: ", xval.shape, yval.shape)
    print("Shape of X and Y test data: ", xtest.shape, ytest.shape)

    xtrain, ytrain = sklearn.utils.shuffle(xtrain, ytrain)
    xval, yval = sklearn.utils.shuffle(xval, yval)
    xtest, ytest = sklearn.utils.shuffle(xtest, ytest)

    # INITIALIZING THE WEIGHT AND BIAS
    model.apply(weights_init)

else: # incase of fusion
    if len(args.input) == 2:

        # LIDAR AND IMAGE
        if 'lidar' in args.input and 'img' in args.input:
            saved_file_name = 'lidar_img'

            # GENERATING THE INPUT AND OUTPUT
            X_lidar_train, X_img_train, ytrain = sklearn.utils.shuffle(X_lidar_train, X_img_train, ytrain)
            X_lidar_validation, X_img_validation, yval = sklearn.utils.shuffle(X_lidar_validation, X_img_validation, yval)
            X_lidar_test, X_img_test, ytest = sklearn.utils.shuffle(X_lidar_test, X_img_test, ytest)


            modelA = LidarNet(input_dim=X_lidar_test.shape[1], output_dim=num_classes, fusion = args.fusion_layer)
            modelB = CameraNet(input_dim=X_img_train.shape[1], output_dim=num_classes, fusion = args.fusion_layer)

            # LOADING BOTH THE MODEL FOR AGGREGATED FUSION
            if args.restore_models:
                lidar_file_name = args.model_folder + '/lidar.pt'
                img_file_name = args.model_folder + '/image.pt'

                tag = ''
                if args.fusion_layer == 'penultimate':
                    tag = tag + "_penultimate"
                    # modelB.load_state_dict(torch.load(coord_file_name)['model_state_dict'])
                    mA = torch.load(lidar_file_name)
                    modelA = torch.nn.Sequential(*(list(mA.children())[:-1]))
                    mB = torch.load(img_file_name)
                    modelB = torch.nn.Sequential(*(list(mB.children())[:-1]))

                    print("modelA:", modelA)
                    print("modelB: ", modelB)
                else:
                    # modelB.load_state_dict(torch.load(coord_file_name)['model_state_dict'])
                    modelA = torch.load(lidar_file_name)
                    modelB = torch.load(img_file_name)
                print("LOADED THE MODELS FOR LIDAR AND IMAGE")

                # FREEZING THE WEIGHTS BEFORE THE FUSION LAYERS
                if args.retrain == False:
                    print("FREEZING THE WEIGHTS BEFORE FUSION LAYERS")
                    for c in modelA.children():
                        for param in c.parameters():
                            param.requires_grad = False
                    for c in modelB.children():
                        for param in c.parameters():
                            param.requires_grad = False

            model = FeatureFusion(modelA, modelB, nb_classes=num_classes, fusion = args.fusion_layer)


    if len(args.input) == 3:
        if 'coord' in args.input and 'img' in args.input and 'lidar' in args.input:
            saved_file_name = 'coord_img_lidar'
            # GENERATING THE INPUT AND OUTPUT
            X_coord_train, X_img_train, X_lidar_train, ytrain = sklearn.utils.shuffle(X_coord_train, X_img_train, X_lidar_train, ytrain)
            X_coord_validation, X_img_validation, X_lidar_validation, yval = sklearn.utils.shuffle(X_coord_validation, X_img_validation, X_lidar_validation, yval)
            X_coord_test, X_img_test, X_lidar_test, ytest = sklearn.utils.shuffle(X_coord_test, X_img_test, X_lidar_test, ytest)


            modelA = CoordNet(input_dim=X_coord_train.shape[1], output_dim=num_classes, fusion = args.fusion_layer)
            modelB = CameraNet(input_dim=X_img_train.shape[1], output_dim=num_classes, fusion = args.fusion_layer)
            modelC = LidarNet(input_dim=X_lidar_train.shape[1], output_dim=num_classes, fusion = args.fusion_layer)

            # LOADING THE MODELS FOR AGGREGATED FUSION
            if args.restore_models:
                coord_file_name = args.model_folder + 'coord_penultimate.pt'
                image_file_name = args.model_folder + 'img_penultimate.pt'
                lidar_file_name = args.model_folder + 'lidar_penultimate.pt'
                tag = ''
                if args.fusion_layer == 'penultimate':
                    tag = tag + "_penultimate"


                    # PENG'S MODEL (ACOUSTIC+SEISMIC) AND DEBASHRI (RADAR) MODEL- NEW SCENARIO BASED SPLIT
                    mA = torch.load(coord_file_name)
                    modelA = torch.nn.Sequential(*(list(mA.children())[:-1]))
                    # modelB.load_state_dict(torch.load(image_file_name)['model_state_dict'])
                    mB = torch.load(image_file_name)
                    modelB = torch.nn.Sequential(*(list(mB.children())[:-1]))
                    mC = torch.load(lidar_file_name)
                    modelC = torch.nn.Sequential(*(list(mC.children())[:-1]))

                    print("modelA:", modelA)
                    print("modelB: ", modelB)
                    print("modelC: ", modelC)
                else:
                    modelA = torch.load(coord_file_name)
                    modelB= torch.load(image_file_name)
                    modelC = torch.load(lidar_file_name)


                # FREEZING THE WEIGHTS BEFORE THE FUSION LAYERS
                if args.retrain == False:
                    print("FREEZING THE WEIGHTS BEFORE FUSION LAYERS")
                    for c in modelA.children():
                        for param in c.parameters():
                            param.requires_grad = False
                    for c in modelB.children():
                        for param in c.parameters():
                            param.requires_grad = False
                    for c in modelC.children():
                        for param in c.parameters():
                            param.requires_grad = False

                print("LOADED THE MODELS FOR COORD, IMAGE AND LIDAR")

            model = InfoFusionThree(modelA, modelB, modelC, nb_classes=num_classes, fusion = args.fusion_layer)
            #model = FeatureFusionThreeMLP(modelA, modelB, modelC, nb_classes=no_of_vehicles, fusion=args.fusion_layer)
            print("Loaded models..")

    # INITIALIZING THE WEIGHT AND BIAS
    model.apply(weights_init)

## WHEN SAVING WEIGHTS FOR PENULTIMATE LAYER
if args.fusion_layer == 'penultimate':  saved_file_name = saved_file_name + "_penultimate"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print("CUDA STATUS: ", use_cuda)
if use_cuda:
    print("CUDA AVAILABLE")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    model.cuda()
else:
    device = torch.device("cpu")

# DATALOADER FOR THREE MODALITY FUSION
class fusion_three_data_loader(object):
    def __init__(self, ds1, ds2, ds3, label):
        self.ds1 = ds1
        self.ds2 = ds2
        self.ds3 = ds3
        self.label = label

    def __getitem__(self, index):
        x1, x2, x3 = self.ds1[index], self.ds2[index],  self.ds3[index]
        label = self.label[index]
        return torch.from_numpy(x1), torch.from_numpy(x2),  torch.from_numpy(x3), torch.from_numpy(label)

    def __len__(self):
        return self.ds1.shape[0]  # assume both datasets have same length


# DATALOADER FOR DUAL FUSION
class fusion_two_data_loader(object):
    def __init__(self, ds1, ds2, label):
        self.ds1 = ds1
        self.ds2 = ds2
        self.label = label

    def __getitem__(self, index):
        x1, x2 = self.ds1[index], self.ds2[index]
        label = self.label[index]
        return torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(label)

    def __len__(self):
        return self.ds1.shape[0]  # assume both datasets have same length

# DATA LOADER FOR SINGLE MODALITY
class data_loader(object):
    def __init__(self, train_val_test):
        if train_val_test == 'train':
            self.feat = xtrain
            self.label = ytrain
        elif train_val_test == 'val':
            self.feat = xval
            self.label = yval
        elif train_val_test == 'test':
            self.feat = xtest
            self.label = ytest
        print(train_val_test)

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, index):
        feat = self.feat[index] #
        label = self.label[index] # change
        return torch.from_numpy(feat), torch.from_numpy(label)



############################################################################################################
#############################    WORKING ON SINGLE MODAL ARCHITECTURES #####################################
############################################################################################################
def single_modal_training(saved_file_name, optimizer_name = 'adam'):

    # loading the data
    # Parameters
    params = {'batch_size': args.bs,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    training_set = data_loader('train')
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = data_loader('val')
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    test_set = data_loader('test')
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    # setting up the loss function
    #pos_weight = torch.as_tensor([2, 3, 3, 2, 2]).to(device) # ADDED FOR WEIGHTED PREFERENCE TO GAS GATOR DURING FUSION
    #pos_weight = torch.ones([5]).to(device)  # Without preferred weight
    # criterion = torch.nn.BCEWithLogitsLoss()  # This loss combines a Sigmoid layer and the BCELoss in one single class.
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.001) # testing for coordinate

    for epoch in range(int(args.epochs)):
        train_correct = 0 # Acc is calculated per epoch for training data
        train_total = 0  # Acc is calculated per epoch for training data
        test_correct = 0
        test_total = 0
        top_k_acc_list = np.zeros(0)  # for top-k accuracies # UPDATE TL BY DR
        total_test_batch = 0  # for top-k accuracies # UPDATE TL BY DR
        model.train()
        # print("Working on epoch ", epoch)
        for train_batch, train_labels in training_generator:
            train_batch, train_labels = train_batch.float().to(device), train_labels.float().to(device)

            outputs = model(train_batch)
            loss = criterion(outputs, torch.max(train_labels, 1)[1])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Getting the Acc
            outputs = outputs.cpu().detach().numpy()
            labels = train_labels.cpu().detach().numpy()
            train_total += labels.shape[0]
            train_correct += (np.argmax(labels, axis=1) == np.argmax(outputs, axis=1)).sum().item()

        model.eval()

        # Test
        for i, (test_batch, test_labels) in enumerate(test_generator): # UPDATE TL BY DR
            test_batch, test_labels = test_batch.float().to(device), test_labels.float().to(device)
            test_output = model(test_batch)

            test_output = test_output.cpu().detach().numpy()
            test_labels = test_labels.squeeze().cpu().detach().numpy()
            print("Sanity check:", np.argmax(test_labels, axis=1), np.argmax(test_output, axis=1))

            # CALCULATING THE TEST ACCURACY
            test_total +=test_labels.shape[0]
            test_correct += (np.argmax(test_labels, axis=1) == np.argmax(test_output, axis= 1)).sum().item()



        test_acc.append(100 * test_correct / test_total)


    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, args.model_folder +"/"+saved_file_name+'.pth')

    torch.save(model, args.model_folder + '/' + saved_file_name + '.pt') # saving the whole model

    # total_train_accuracy = train_accuracy.compute()
    # print("Total train accuracy..", total_train_accuracy)

    # print("Total and Correct estimates: ", test_total, test_correct)
    # return (100 * test_correct / test_total)

if fusion is False and "coord" in args.input:

    single_modal_training(saved_file_name,'adam')
    print("Test Accuracies: ", test_acc)
    print("Final test accuracy: ", test_acc[int(args.epochs)-1])
    print("End of Coordinate")

if fusion is False and "img" in args.input:

    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay = 1)
    single_modal_training(saved_file_name, 'adam')
    print("Test Accuracies: ", test_acc)
    print("Final test accuracy: ", test_acc[int(args.epochs)-1])
    print("End of Image")

if fusion is False and "lidar" in args.input:
    single_modal_training(saved_file_name, 'adam')
    print("Test Accuracies: ", test_acc)
    print("Final test accuracy: ", test_acc[int(args.epochs)-1])
    print("End of Lidar")


############################################################################################################
#############################    WORKING ON FUSION ARCHITECTURES #####################################
############################################################################################################
def two_modality_training(saved_file_name, xtrain_mod1, xtrain_mod2, ytrain, xval_mod1, xval_mod2, yval, xtest_mod1, xtest_mod2, ytest):
    params = {'batch_size': args.bs,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    training_set = fusion_two_data_loader(xtrain_mod1, xtrain_mod2, ytrain)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = fusion_two_data_loader(xval_mod1, xval_mod2, yval)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    testing_set = fusion_two_data_loader(xtest_mod1, xtest_mod2, ytest)
    test_generator = torch.utils.data.DataLoader(testing_set, **params)
    #model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # Loss and optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)


    for epoch in range(int(args.epochs)):
        train_correct = 0 # Acc is calculated per epoch for training data
        train_total = 0  # Acc is calculated per epoch for training data
        test_correct = 0
        test_total = 0
        top_k_acc_list = np.zeros(0)  # for top-k accuracies # UPDATE TL BY DR
        total_test_batch = 0  # for top-k accuracies # UPDATE TL BY DR
        for i, (batch1, batch2, train_labels) in enumerate(training_generator):
                batch1, batch2, train_labels = batch1.float().to(device), batch2.float().to(device), train_labels.float().to(device)

                # Forward pass
                outputs, hidden_layers = model(batch1, batch2)
                loss = criterion(outputs, torch.max(train_labels, 1)[1])


                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Getting Acc
                outputs = outputs.cpu().detach().numpy()
                labels = train_labels.cpu().detach().numpy()
                train_total += labels.shape[0]
                train_correct += (np.argmax(labels, axis=1) == np.argmax(outputs, axis=1)).sum().item()
        model.eval()

        # Test
        for i, (test_batch1, test_batch2, test_labels) in enumerate(test_generator): # UPDATE TL BY DR
            test_batch1, test_batch2, test_labels = test_batch1.float().to(device), test_batch2.float().to(
                device), test_labels.float().to(device)

            test_output, _ = model(test_batch1, test_batch2)

            test_output = test_output.cpu().detach().numpy()
            test_labels = test_labels.squeeze().cpu().detach().numpy()
            print("Sanity check:", np.argmax(test_labels, axis=1), np.argmax(test_output, axis=1))

            # CALCULATING THE TEST ACCURACY
            test_total += test_labels.shape[0]
            test_correct += (np.argmax(test_labels, axis=1) == np.argmax(test_output, axis=1)).sum().item()
        test_acc.append(100 * test_correct / test_total)




    torch.save(model, args.model_folder + '/' + saved_file_name + '.pt')
    if args.retrain: torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, args.model_folder + '/' + saved_file_name + '.pth')

    # print("Total and Correct estimates: ", test_total, test_correct)
    # return (100 * test_correct / test_total)


# Fusion between image and coordinate
if fusion is True and len(args.input) == 2 and 'lidar' in args.input and 'img' in args.input:
    two_modality_training(saved_file_name, X_lidar_train, X_img_train, ytrain, X_lidar_validation, X_img_validation,
                                     yval, X_lidar_test, X_img_test, ytest)
    print("Test Accuracies: ", test_acc)
    print("Final test accuracy: ", test_acc[int(args.epochs)-1])
    print("End of LiDAR and Image Fusion")



def three_modality_training(saved_file_name, xtrain_mod1, xtrain_mod2, xtrain_mod3, ytrain, xval_mod1, xval_mod2, xval_mod3, yval, xtest_mod1, xtest_mod2, xtest_mod3, ytest):
    params = {'batch_size': args.bs,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    training_set = fusion_three_data_loader(xtrain_mod1, xtrain_mod2, xtrain_mod3, ytrain)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = fusion_three_data_loader(xval_mod1, xval_mod2, xval_mod3, yval)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    testing_set = fusion_three_data_loader(xtest_mod1, xtest_mod2, xtest_mod3, ytest)
    test_generator = torch.utils.data.DataLoader(testing_set, **params)

    criterion = torch.nn.CrossEntropyLoss()
    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)


    for epoch in range(int(args.epochs)):
        train_correct = 0 # Acc is calculated per epoch for training data
        train_total = 0  # Acc is calculated per epoch for training data
        test_correct = 0
        test_total = 0
        top_k_acc_list =np.zeros(0) # for top-k accuracies # UPDATE TL BY DR
        total_test_batch = 0 # for top-k accuracies # UPDATE TL BY DR
        for i, (batch1, batch2, batch3, train_labels) in enumerate(training_generator):
                batch1, batch2, batch3, train_labels = batch1.float().to(device), batch2.float().to(device), batch3.float().to(device), train_labels.float().to(device)


                # Forward pass
                outputs = model(batch1, batch2, batch3)
                loss = criterion(outputs, torch.max(train_labels, 1)[1])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #Getting the Acc
                outputs = outputs.cpu().detach().numpy()
                labels = train_labels.cpu().detach().numpy()
                train_total += labels.shape[0]
                train_correct += (np.argmax(labels, axis=1) == np.argmax(outputs, axis=1)).sum().item()

        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

        # Test
        for i, (test_batch1, test_batch2, test_batch3, test_labels) in enumerate(test_generator): # UPDATE TL BY DR
            test_batch1, test_batch2, test_batch3, test_labels = test_batch1.float().to(device), test_batch2.float().to(device), test_batch3.float().to(device), test_labels.float().to(device)
            test_output = model(test_batch1, test_batch2, test_batch3)


            test_output = test_output.cpu().detach().numpy()
            test_labels = test_labels.squeeze().cpu().detach().numpy()
            print("Sanity check:", np.argmax(test_labels, axis=1), np.argmax(test_output, axis=1))

            # CALCULATING THE TEST ACCURACY
            test_total += test_labels.shape[0]
            test_correct += (np.argmax(test_labels, axis=1) == np.argmax(test_output, axis=1)).sum().item()


        test_acc.append(100 * test_correct / test_total)


        # print loss and accuracies


    if args.retrain:
        torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, args.model_folder + '/' + saved_file_name + '.pth')

    torch.save(model, args.model_folder + '/' + saved_file_name + '.pt') # saving the whole model


if fusion is True and len(args.input) == 3 and 'coord' in args.input and 'img' in args.input and 'lidar' in args.input:
    coord_file_name = args.model_folder + 'coord.pt'
    img_file_name = args.model_folder + 'img.pt'
    lidar_file_name = args.model_folder + 'lidar.pt'
    fusion_file_name = args.model_folder + 'coord_img_lidar_penultimate.pt'

    three_modality_training(saved_file_name, X_coord_train, X_img_train, X_lidar_train, ytrain, X_coord_validation, X_img_validation, X_lidar_validation, yval, X_coord_test, X_img_test, X_lidar_test, ytest)

    print("Test Accuracies: ", test_acc)
    print("Final test accuracy: ", test_acc[int(args.epochs) - 1])
    print("End of Coordinate, Image and LiDAR Fusion")

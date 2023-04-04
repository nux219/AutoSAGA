#Code for AutoSAGA attack
import sys
import os

from self_models import *
import DataManagerPytorch as DMP
import ModelPlus
from transfomer_models.TransformerModels import VisionTransformer, CONFIGS
from transfomer_models import BigTransferModels
from collections import OrderedDict  # pylint: disable=g-importing-member
from snn_models import resnet_spiking_SAGA, vgg_spiking_imagenet, SNNBackpropModel, vgg_spiking_SAGA
from snn_models import snn_functions, SpikingJellyResNet, SpikingJelly_spiking_resnet as spiking_resnet, SpikingJelly_sew_resnet as sew_resnet
from spikingjelly.clock_driven import neuron, surrogate
import AttackWrappersWhiteBoxP_SAGA
import AttackWrappersWhiteBoxSNN
import AttackWrappersWhiteBoxJelly
import AttackWrappersProtoSAGA

def SNN_SAGA(modelDir, syntheticmodelDir, dataset, secondcoeff):
    saveTag ="SAGA Attack SNN"
    device = torch.device("cuda")
    # dataset = 'CIFAR10'
    imgSize = 32
    # batchSize = 64
    batchSize = 128
    if dataset == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        # mean = (0.5)
        # std = (0.5)
        numClasses = 10
        valLoader = DMP.GetCIFAR10Validation_norm(imgSize, batchSize, mean=0, std=1)

    elif dataset == 'CIFAR100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        numClasses = 100
        valLoader = DMP.GetCIFAR100Validation_norm(imgSize, batchSize, mean, std)
    #Load the defense
    # defense = Loadresnet(modelDir)
    modelPlusList = []
    defense, ModelPlus_SNN, SNN_model = LoadDietSNN(modelDir, dataset, mean, std)
    modelPlusList.append(ModelPlus_SNN)
    # defense, ModelPlus_defense, SNN_model = LoadDietSNN_resnet(modelDir)
    # print('\n {}'.format(SNN_model))
    # defense = LoadBARZ8(modelDir)
    #Attack parameters
    numAttackSamples = 1000
    print('numAttackSamples: ', numAttackSamples)
    # epsForAttacks = 0.05
    epsForAttacks = 0.031
    # epsForAttacks = 0.062
    clipMin = 0.0
    clipMax = 1.0
    # clipMin = -1.0
    # clipMax = 1.0

    # valLoader = DMP.GetCIFAR10Validation_unnormalize(imgSize, batchSize, mean=0.5)
    cleanAcc = defense.validateD(valLoader)
    print("SNN Accuracy: ", cleanAcc)
    # sys.exit()
    #Get the clean data
    # xTest, yTest = DMP.DataLoaderToTensor(valLoader)
    # cleanLoader = DMP.TensorToDataLoader(xTest[0:1000], yTest[0:1000], transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    #Create the synthetic model
    # syntheticModel = NetworkConstructorsAdaptive.CarliniNetwork(imgSize, numClasses)
    # # ================================Load CNN models==============================================
    # # syntheticModel = resnet56.resnet56(32, numClasses)
    # syntheticModel = VGG(vgg_name='VGG16', labels=numClasses, mean=mean, std=std)
    # # syntheticModel = ResNet20(labels=numClasses, dropout=0.2)
    # # syntheticModel = nn.DataParallel(syntheticModel)
    # checkpoint = torch.load(syntheticmodelDir)
    # syntheticModel.load_state_dict(checkpoint['state_dict'])
    # syntheticModel.to(device)
    # syntheticModel.eval()
    # ModelPlus_CNN = ModelPlus.ModelPlus("VGG16", syntheticModel, device, imgSize, imgSize, batchSize)
    # synAcc = DMP.validateD(valLoader, syntheticModel)
    # print("CNN Accuracy: ", synAcc)
    # # ================================Load CNN models==============================================
    # syntheticModel, syntheticModelPlus, modelPlusList, valLoader_unnormalize = LoadBiT_R101()
    # syntheticModel, syntheticModelPlus, modelPlusList, valLoader_unnormalize = LoadBiT_R50()
    # syntheticModel, syntheticModelPlus, modelPlusList, valLoader_unnormalize = LoadViTL()
    syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_unnormalize = LoadViTB()
    syntheticModel.to(device)
    syntheticModel.eval()
    modelPlusList.append(ModelPlus_CNN)

    SNN_model.to(device)
    synAcc = DMP.validateD(valLoader_unnormalize, syntheticModel)
    # synAcc = DMP.validateD(valLoader, syntheticModel)
    print("CNN Accuracy: ", synAcc)

    cleanAccA = ModelPlus_CNN.validateD(valLoader)
    cleanAccB = ModelPlus_SNN.validateD(valLoader_unnormalize)
    print("CNN Accuracy: ", cleanAccA)
    print("SNN Accuracy: ", cleanAccB)

    # advLoaderFGSM = AttackWrappersWhiteBoxP_ori.FGSMNativePytorch(device, valLoader, SNN_model, epsForAttacks, clipMin, clipMax, targeted = False)
    # cleanLoader, cleanDataLoader_B = DMP.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader,
    #                                                               ModelPlus_SNN, ModelPlus_CNN)
    # cleanLoader = DMP.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader,
    #                                                               ModelPlus_SNN, ModelPlus_CNN)
    cleanLoader, cleanDataLoader_B = DMP.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader_unnormalize,
                                                                  ModelPlus_CNN, ModelPlus_SNN)
    # cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, numAttackSamples, valLoader, numClasses)
    # SNNAcc_clean = DMP.validateD(cleanLoader, SNN_model)
    # print("SNN Model cleanLoader Accuracy: ", SNNAcc_clean)
    #Do the attack
    oracle = defense
    # dataLoaderForTraining = trainLoader
    optimizerName = "adam"
    #Last line does the attack
    decayFactor = 1.0
    # numSteps = 40
    numSteps = 10
    # numSteps = 7
    # epsStep = epsForAttacks/numSteps
    # epsStep = 0.005
    epsStep = 0.01
    print('numSteps: ', numSteps, 'epsStep: ',epsStep, 'epsForAttacks: ', epsForAttacks)
    coefficientArray = torch.zeros(2)
    # secondcoeff = 2.0000e-04
    # secondcoeff = 0
    coefficientArray[0] = 1.0 - secondcoeff
    coefficientArray[1] = secondcoeff
    print("Coeff Array:")
    print(coefficientArray)

    advLoader = AttackWrappersWhiteBoxP_SAGA.SelfAttentionGradientAttack(device, epsForAttacks, numSteps, epsStep,
                                                modelPlusList, coefficientArray, cleanLoader, clipMin, clipMax, mean, std)

    # torch.save(advLoaderMIM, saveDir+"//AdvLoaderMIM")
    torch.cuda.empty_cache()

    print("SNN Accuracy: ", cleanAcc)
    print("CNN Accuracy: ", synAcc)
    print("Coeff Array:")
    print(coefficientArray)
    #Go through and check the robust accuray of each model on the adversarial examples
    for i in range(0, len(modelPlusList)):
        acc = modelPlusList[i].validateD(advLoader)
        print(modelPlusList[i].modelName+" Robust Acc:", acc)


def SNN_AutoSAGA_two(modelDir1, modelDir2, dataset):
    saveTag ="AutoSAGA Attack SNN"
    device = torch.device("cuda")
    # dataset = 'CIFAR10'

    # batchSize = 64
    batchSize = 128
    # batchSize = 32
    if dataset == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)     # set mean for dataset
        std = (0.2023, 0.1994, 0.2010)      # set std for dataset
        imgSize = 32
        # mean = (0.5)
        # std = (0.5)
        numClasses = 10
        valLoader = DMP.GetCIFAR10Validation_norm(imgSize, batchSize, mean=0, std=1)

    elif dataset == 'CIFAR100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        numClasses = 100
        imgSize = 32
        valLoader = DMP.GetCIFAR100Validation_norm(imgSize, batchSize, mean=0, std=1)
    elif dataset == 'IMAGENET':
        numClasses = 1000
        imgSize = 224
        valLoader = DMP.GetIMAGENETValidation(imgSize, batchSize)
    #Load the defense
    # defense = Loadresnet(modelDir)
    modelPlusList = []

    #Create the synthetic model
    # # ================================Load CNN models==============================================
    # syntheticModel = resnet56.resnet56(32, numClasses)
    syntheticModel = VGG(vgg_name='VGG16', labels=numClasses, mean=mean, std=std)
    #ResNet-152
    # syntheticModel = ResNetPretrainedImageNet.resnet152(pretrained=True)
    # syntheticModel = ResNet20(labels=numClasses, dropout=0.2)
    # syntheticModel = nn.DataParallel(syntheticModel)
    checkpoint = torch.load(modelDir2)
    syntheticModel.load_state_dict(checkpoint['state_dict'])
    syntheticModel.to(device)
    syntheticModel.eval()
    ModelPlus_CNN = ModelPlus.ModelPlus("VGG16", syntheticModel, device, imgSize, imgSize, batchSize)
    # ModelPlus_CNN = ModelPlus.ModelPlus("ResNet-152", syntheticModel, device, imgSize, imgSize, batchSize)
    modelPlusList.append(ModelPlus_CNN)
    synAcc = DMP.validateD(valLoader, syntheticModel)
    print("CNN Accuracy: ", synAcc)
    # # ================================Load CNN models==============================================

    # # ================================Load Transformer models==============================================
    valLoader_resized = None
    # # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadBiT_R152()
    # # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadBiT_R101()
    # # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadBiT_R50()
    # # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadViTL_imagenet()
    # # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadViTL()
    # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadViTB()
    # # defense_CNN, ModelPlus_CNN, syntheticModel = LoadDietSNN(modelDir2, dataset, batchSize, mean, std, timesteps=20)
    # syntheticModel.to(device)
    # syntheticModel.eval()
    # modelPlusList.append(ModelPlus_CNN)
    # synAcc = DMP.validateD(valLoader, syntheticModel)
    # # synAcc = DMP.validateD(valLoader_resized, syntheticModel)
    # print("CNN Accuracy: ", synAcc)
    # # ================================Load Transformer models==============================================
    # # ================================Load model 1==============================================

    # # ================================Load SNN models==============================================
    ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN(modelDir1, dataset, batchSize, mean, std, timesteps=5)
    # ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN_resnet(modelDir1, dataset, batchSize, mean, std)
    modelPlusList.append(ModelPlus_SNN_vgg)
    SNN_model_vgg.to(device)
    transvggAcc = DMP.validateD(valLoader, SNN_model_vgg)
    print("trans snn Accuracy: ", transvggAcc)

    # defense_res, ModelPlus_SNN_res, SNN_model_res = LoadDietSNN_resnet(modelDir1, dataset, batchSize, mean, std)
    # modelPlusList.append(ModelPlus_SNN_res)
    # SNN_model_res.to(device)
    # transresAcc = DMP.validateD(valLoader, SNN_model_res)
    # print("trans resnet Accuracy: ", transresAcc)
    # # ================================Load SNN models==============================================
    # # ================================Load model 2==============================================
    numModels = 2
    totalSampleNum = len(valLoader.dataset)
    # Get accuracy array for each model
    accArrayCumulative = torch.zeros(totalSampleNum).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(valLoader)
        accArrayCumulative = accArrayCumulative + accArray
    MV_clean_acc = (accArrayCumulative==2).sum() / totalSampleNum
    print('All_clean_acc: ', MV_clean_acc.data.cpu().numpy())
    MV_clean_acc = (accArrayCumulative>0).sum() / totalSampleNum
    print('clean_acc: ', MV_clean_acc.data.cpu().numpy())

    #Attack parameters
    numAttackSamples = 1000
    print('numAttackSamples: ', numAttackSamples)

    # # ================================Get clean data that is correct on both models==============================================
    if valLoader_resized is None:
        cleanLoader = AttackWrappersProtoSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader,
                                                                  modelPlusList)
    else:
        cleanLoader = AttackWrappersProtoSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader_resized,
                                                                  modelPlusList)


    # cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, numAttackSamples, valLoader, numClasses)
    # SNNAcc_clean = DMP.validateD(cleanLoader, SNN_model)
    # print("SNN Model cleanLoader Accuracy: ", SNNAcc_clean)
    # (cleanData, cleanTarget) = torch.load('./Clean_Sun Aug 14 01:25:44 2022.pt')
    # cleanLoader = DMP.TensorToDataLoader(cleanData, cleanTarget, transforms=None, batchSize=10,
    #                                            randomizer=None)
    # # ================================Get clean data that is correct on both models==============================================

    # # ================================Do the attack==============================================

    # # ======================================Auto SAGA attack=====================================================
    # dataLoaderForTraining = trainLoader
    # epsForAttacks = 0.05
    epsMax = 0.031
    # epsForAttacks = 0.062
    clipMin = 0.0
    clipMax = 1.0
    # clipMin = -1.0
    # clipMax = 1.0
    decayFactor = 1.0
    numSteps = 25
    # numSteps = 10
    # numSteps = 7
    # epsStep = epsForAttacks/numSteps
    epsStep = 0.005
    # epsStep = 0.01
    print('numSteps: ', numSteps, 'epsStep: ',epsStep, 'epsMax: ', epsMax)

    alphaLearningRate = 10000#0#100000
    # alphaLearningRate = 100000#0#100000
    # alphaLearningRate = 10000#0#100000
    fittingFactor = 50.0
    print("Alpha Learning Rate:", alphaLearningRate)
    print("Fitting Factor:", fittingFactor)

    torch.cuda.empty_cache()
    advLoader = AttackWrappersProtoSAGA.SelfAttentionGradientAttackProto(device, epsMax, epsStep, numSteps, modelPlusList,
                                                    cleanLoader, clipMin, clipMax, alphaLearningRate, fittingFactor,advLoader=None, numClasses=numClasses)

    accArrayProtoSAGA = torch.zeros(numAttackSamples).to(device).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoader)
        accArrayProtoSAGA = accArrayProtoSAGA + accArray
        print("ProtoSAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_ProtoSAGA_acc = (accArrayProtoSAGA>=1).sum() / numAttackSamples
    print('MV_ProtoSAGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    ALL_MV_ProtoSAGA_acc = (accArrayProtoSAGA==2).sum() / numAttackSamples
    print('ALL_MV_ProtoSAGA_acc: ', ALL_MV_ProtoSAGA_acc.data.cpu().numpy())
    MV_ProtoSAGA_acc = (accArrayProtoSAGA==0).sum() / numAttackSamples
    print('ProtoSAGA attack successful rate: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    # torch.save(advLoaderMIM, saveDir+"//AdvLoaderMIM")
    torch.cuda.empty_cache()
    # sys.exit()


    # # ======================================Basic SAGA attack=====================================================

    print('Basic SAGA attack')
    coefficientArray = torch.zeros(2)
    # secondcoeff = 2.0000e-04
    secondcoeff = 0.5       # set coefficent for SAGA attack that balance the impact of the two models
    coefficientArray[0] = 1.0 - secondcoeff
    coefficientArray[1] = secondcoeff
    print("Coeff Array:")
    print(coefficientArray)
    advLoader = AttackWrappersWhiteBoxP_SAGA.SelfAttentionGradientAttack(device, epsMax, numSteps, epsStep,
                                                modelPlusList, coefficientArray, cleanLoader, clipMin, clipMax, mean, std)

    accArraySAGA = torch.zeros(numAttackSamples).to(device).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoader)
        accArraySAGA = accArraySAGA + accArray
        print("SAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_ProtoSAGA_acc = (accArraySAGA>=1).sum() / numAttackSamples
    print('MV_AGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    ALL_MV_SAGA_acc = (accArraySAGA>1).sum() / numAttackSamples
    print('ALL_MV_SAGA_acc: ', ALL_MV_SAGA_acc.data.cpu().numpy())
    MV_SAGA_acc = (accArraySAGA==0).sum() / numAttackSamples
    print('SAGA attack successful rate: ', MV_SAGA_acc.data.cpu().numpy())

    # # ====================================== MIM attack for each model =====================================================
    print()
    numSteps = 40
    epsStep = 0.01
    print('numSteps: ', numSteps, 'epsStep: ', epsStep, 'epsMax: ', epsMax)
    print()
    print('MIM attack')
    advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN, decayFactor, epsMax, epsStep,
                                                            numSteps, clipMin, clipMax, targeted=False)
    accArrayMIM_A = torch.zeros(numAttackSamples).to(device).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderMIM)
        accArrayMIM_A = accArrayMIM_A + accArray
        print("MIM_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_MIM_A_acc = (accArrayMIM_A>=1).sum() / numAttackSamples
    print('MIM_A_acc: ', MV_MIM_A_acc.data.cpu().numpy())
    ALL_MV_MIM_A_acc = (accArrayMIM_A>1).sum() / numAttackSamples
    print('ALL_MIM_A_acc: ', ALL_MV_MIM_A_acc.data.cpu().numpy())
    ALL_MV_MIM_A_acc = (accArrayMIM_A==0).sum() / numAttackSamples
    print('MIM_A attack successful rate: ', ALL_MV_MIM_A_acc.data.cpu().numpy())
    torch.cuda.empty_cache()

    advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, decayFactor, epsMax, epsStep,
                                                            numSteps, clipMin, clipMax, targeted=False)
    accArrayMIM_B = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderMIM)
        accArrayMIM_B = accArrayMIM_B + accArray
        print("MIM_B Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_MIM_B_acc = (accArrayMIM_B>=1).sum() / numAttackSamples
    print('MV_MIM_B_acc: ', MV_MIM_B_acc.data.cpu().numpy())
    ALL_MV_MIM_B_acc = (accArrayMIM_B>1).sum() / numAttackSamples
    print('ALL_MV_MIM_B_acc: ', ALL_MV_MIM_B_acc.data.cpu().numpy())
    ALL_MV_MIM_B_acc = (accArrayMIM_B==0).sum() / numAttackSamples
    print('MIM_B attack successful rate: ', ALL_MV_MIM_B_acc.data.cpu().numpy())
    torch.cuda.empty_cache()

    # # ====================================== PGD attack for each model =====================================================
    print('PGD attack')
    advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    accArrayPGD_A = torch.zeros(numAttackSamples).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_A = accArrayPGD_A + accArray
        print("PGD_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_PGD_A_acc = (accArrayPGD_A>=1).sum() / numAttackSamples
    print('MV_PGD_A_acc: ', MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A>1).sum() / numAttackSamples
    print('ALL_MV_PGD_A_acc: ', ALL_MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A==0).sum() / numAttackSamples
    print('PGD_A attack successful rate: ', ALL_MV_PGD_A_acc.data.cpu().numpy())

    advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    accArrayPGD_B = torch.zeros(numAttackSamples).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_B = accArrayPGD_B + accArray
        print("PGD_B Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_PGD_B_acc = (accArrayPGD_B>=1).sum() / numAttackSamples
    print('MV_PGD_B_acc: ', MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B>1).sum() / numAttackSamples
    print('ALL_MV_PGD_B_acc: ', ALL_MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B==0).sum() / numAttackSamples
    print('PGD_B attack successful rate: ', ALL_MV_PGD_B_acc.data.cpu().numpy())

    # # ====================================== Auto PGD attack for each model =====================================================
    etaStart = 0.05
    numSteps = 40
    epsStep = 0.01
    print('numSteps: ', numSteps, 'epsStep: ', epsStep, 'epsMax: ', epsMax)
    print('Auto Attack')
    advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.AutoAttackNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN,
                                                           epsMax, etaStart,
                                                           numSteps, clipMin, clipMax, targeted=False)
    accArrayPGD_A = torch.zeros(numAttackSamples).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_A = accArrayPGD_A + accArray
        print("Auto_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_PGD_A_acc = (accArrayPGD_A>=1).sum() / numAttackSamples
    print('MV_Auto_A_acc: ', MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A>1).sum() / numAttackSamples
    print('ALL_MV_Auto_A_acc: ', ALL_MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A==0).sum() / numAttackSamples
    print('Auto_A attack successful rate: ', ALL_MV_PGD_A_acc.data.cpu().numpy())

    advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.AutoAttackNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, epsMax,
                                                         etaStart,
                                                         numSteps, clipMin, clipMax, targeted=False)
    accArrayPGD_B = torch.zeros(numAttackSamples).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_B = accArrayPGD_B + accArray
        print("Auto_B Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_PGD_B_acc = (accArrayPGD_B>=1).sum() / numAttackSamples
    print('MV_Auto_B_acc: ', MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B>1).sum() / numAttackSamples
    print('ALL_MV_Auto_B_acc: ', ALL_MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B==0).sum() / numAttackSamples
    print('Auto_B attack successful rate: ', ALL_MV_PGD_B_acc.data.cpu().numpy())



def SNN_AutoSAGA_two_snnsnn(modelDir1, modelDir2, dataset):
    saveTag ="AutoSAGA Attack SNN"
    device = torch.device("cuda")
    # dataset = 'CIFAR10'
    imgSize = 32
    batchSize = 100
    # batchSize = 32
    if dataset == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        # mean = (0.5)
        # std = (0.5)
        numClasses = 10
        valLoader = DMP.GetCIFAR10Validation_norm(imgSize, batchSize, mean=0, std=1)

    elif dataset == 'CIFAR100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        numClasses = 100
        valLoader = DMP.GetCIFAR100Validation_norm(imgSize, batchSize, mean, std)
    #Load the defense
    # defense = Loadresnet(modelDir)
    modelPlusList = []

    #Create the synthetic model
    # syntheticModel = NetworkConstructorsAdaptive.CarliniNetwork(imgSize, numClasses)
    # # ================================Load CNN models==============================================
    # # syntheticModel = resnet56.resnet56(32, numClasses)
    # syntheticModel = VGG(vgg_name='VGG16', labels=numClasses, mean=mean, std=std)
    # # syntheticModel = ResNet20(labels=numClasses, dropout=0.2)
    # # syntheticModel = nn.DataParallel(syntheticModel)
    # checkpoint = torch.load(modelDir2)
    # syntheticModel.load_state_dict(checkpoint['state_dict'])
    # syntheticModel.to(device)
    # syntheticModel.eval()
    # ModelPlus_CNN = ModelPlus.ModelPlus("VGG16", syntheticModel, device, imgSize, imgSize, batchSize)
    # modelPlusList.append(ModelPlus_CNN)
    # synAcc = DMP.validateD(valLoader, syntheticModel)
    # print('load CNN from: ', modelDir2)
    # print("CNN Accuracy: ", synAcc)
    # # ================================Load CNN models==============================================

    valLoader_resized = None
    # # ================================Load ViT models==============================================
    # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadBiT_R101()
    # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadBiT_R50()
    # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadViTL()
    # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadViTB()
    # syntheticModel.to(device)
    # syntheticModel.eval()
    # modelPlusList.append(ModelPlus_CNN)
    # synAcc = DMP.validateD(valLoader_resized, syntheticModel)
    # print("CNN Accuracy: ", synAcc)
    # # ================================Load ViT models==============================================
    ## ====================load RESNET BP SNN=================================================
    syntheticModel = SpikingJellyResNet.LoadCIFAR10SNNResNetBackProp("./resnet18acc_8183.pt")
    print('load snn resent bp ./resnet18acc_8183.pt' )
    ModelPlus_CNN = ModelPlus.ModelPlusSpikingJelly("SNN ResNet Backprop", syntheticModel, device, 32, 32, batchSize)
    syntheticModel.to(device)
    syntheticModel.eval()
    modelPlusList.append(ModelPlus_CNN)
    synAcc = ModelPlus_CNN.validateD(valLoader)
    print("SNN ResNet Backprop Accuracy: ", synAcc)
    ## ====================load RESNET BP SNN=================================================

    ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN(modelDir1, dataset, batchSize, mean, std)
    # ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN_resnet(modelDir1, dataset, batchSize, mean, std)
    # SNN_model_vgg, ModelPlus_SNN_vgg, valLoader_repeated = LoadVGG16SNN_bp(batchSize=32)

    # # ====================load RESNET BP SNN=================================================
    # SNN_model_vgg = SpikingJellyResNet.LoadCIFAR10SNNResNetBackProp("./resnet18acc_8183.pt")
    # print('load snn resent bp ./resnet18acc_8183.pt' )
    # SNN_model_vgg.eval()
    # ModelPlus_SNN_vgg = ModelPlus.ModelPlusSpikingJelly("SNN ResNet Backprop", SNN_model_vgg, device, 32, 32, batchSize)
    # # ====================load RESNET BP SNN=================================================

    modelPlusList.append(ModelPlus_SNN_vgg)
    SNN_model_vgg.to(device)
    # transvggAcc = DMP.validateD(valLoader, SNN_model_vgg)
    transvggAcc = ModelPlus_SNN_vgg.validateD(valLoader)
    # transvggAcc = DMP.validateD(valLoader_repeated, SNN_model_vgg)
    print("trans snn Accuracy: ", transvggAcc)

    # defense_res, ModelPlus_SNN_res, SNN_model_res = LoadDietSNN_resnet(modelDir1, dataset, batchSize, mean, std)
    # modelPlusList.append(ModelPlus_SNN_res)
    # SNN_model_res.to(device)
    # transresAcc = DMP.validateD(valLoader, SNN_model_res)
    # print("trans resnet Accuracy: ", transresAcc)

    # xTest, yTest = DMP.DataLoaderToTensor(valLoader_repeated)
    numModels = 2
    totalSampleNum = len(valLoader.dataset)
    # Get accuracy array for each model
    accArrayCumulative = torch.zeros(totalSampleNum).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(valLoader)
        accArrayCumulative = accArrayCumulative + accArray
    MV_clean_acc = (accArrayCumulative==2).sum() / totalSampleNum
    print('All_clean_acc: ', MV_clean_acc.data.cpu().numpy())
    MV_clean_acc = (accArrayCumulative>0).sum() / totalSampleNum
    print('clean_acc: ', MV_clean_acc.data.cpu().numpy())

    #Attack parameters
    numAttackSamples = 1000
    print('numAttackSamples: ', numAttackSamples)

    if valLoader_resized is None:
        cleanLoader = AttackWrappersProtoSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader,
                                                                  modelPlusList)
    else:
        cleanLoader = AttackWrappersProtoSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader_resized,
                                                                  modelPlusList)

    # (cleanData, cleanTarget) = torch.load('./Clean_Sun Aug 14 13:47:45 2022.pt')
    # cleanLoader = DMP.TensorToDataLoader(cleanData, cleanTarget, transforms=None, batchSize=10,
    #                                            randomizer=None)

    #Do the attack
    # dataLoaderForTraining = trainLoader
    # epsForAttacks = 0.05
    epsMax = 0.031
    # epsForAttacks = 0.062
    clipMin = 0.0
    clipMax = 1.0
    # clipMin = -1.0
    # clipMax = 1.0
    decayFactor = 1.0
    numSteps = 40
    # numSteps = 80
    # numSteps = 7
    # epsStep = epsForAttacks/numSteps
    epsStep = 0.005
    # epsStep = 0.01
    print('numSteps: ', numSteps, 'epsStep: ',epsStep, 'epsMax: ', epsMax)

    alphaLearningRate = 10000#0#100000
    # alphaLearningRate = 100000#0#100000
    # alphaLearningRate = 10000#0#100000
    fittingFactor = 50.0
    print("Alpha Learning Rate:", alphaLearningRate)
    print("Fitting Factor:", fittingFactor)

    torch.cuda.empty_cache()
    advLoader = AttackWrappersProtoSAGA.SelfAttentionGradientAttackProto(device, epsMax, epsStep, numSteps, modelPlusList,
                                                    cleanLoader, clipMin, clipMax, alphaLearningRate, fittingFactor,numClasses=numClasses)
    accArrayProtoSAGA = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoader)
        accArrayProtoSAGA = accArrayProtoSAGA + accArray
        print("ProtoSAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_ProtoSAGA_acc = (accArrayProtoSAGA>=1).sum() / numAttackSamples
    print('MV_ProtoSAGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    ALL_MV_ProtoSAGA_acc = (accArrayProtoSAGA==2).sum() / numAttackSamples
    print('ALL_MV_ProtoSAGA_acc: ', ALL_MV_ProtoSAGA_acc.data.cpu().numpy())
    MV_ProtoSAGA_acc = (accArrayProtoSAGA==0).sum() / numAttackSamples
    print('ProtoSAGA attack successful rate: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    # torch.save(advLoaderMIM, saveDir+"//AdvLoaderMIM")
    torch.cuda.empty_cache()


    print()
    print('Basic SAGA attack')
    coefficientArray = torch.zeros(2)
    # secondcoeff = 2.0000e-04
    secondcoeff = 0.5
    coefficientArray[0] = 1.0 - secondcoeff
    coefficientArray[1] = secondcoeff
    print("Coeff Array:")
    print(coefficientArray)
    advLoader = AttackWrappersWhiteBoxP_SAGA.SelfAttentionGradientAttack(device, epsMax, numSteps, epsStep,
                                                modelPlusList, coefficientArray, cleanLoader, clipMin, clipMax, mean, std)
    accArraySAGA = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoader)
        accArraySAGA = accArraySAGA + accArray
        print("SAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_ProtoSAGA_acc = (accArraySAGA>=1).sum() / numAttackSamples
    print('MV_AGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    ALL_MV_SAGA_acc = (accArraySAGA>1).sum() / numAttackSamples
    print('ALL_MV_SAGA_acc: ', ALL_MV_SAGA_acc.data.cpu().numpy())
    MV_SAGA_acc = (accArraySAGA==0).sum() / numAttackSamples
    print('SAGA attack successful rate: ', MV_SAGA_acc.data.cpu().numpy())


    print()
    print('MIM attack')
    if ModelPlus_CNN.modelName=='SNN ResNet Backprop':
        advLoaderMIM = AttackWrappersWhiteBoxJelly.MIMNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN,
                                                                  decayFactor, epsMax, epsStep, numSteps, clipMin,
                                                                  clipMax, targeted=False)
    else:
        advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN, decayFactor, epsMax, epsStep,
                                                            numSteps, clipMin, clipMax, targeted=False)
    accArrayMIM_A = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderMIM)
        accArrayMIM_A = accArrayMIM_A + accArray
        print("MIM_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_MIM_A_acc = (accArrayMIM_A>=1).sum() / numAttackSamples
    print('MIM_A_acc: ', MV_MIM_A_acc.data.cpu().numpy())
    ALL_MV_MIM_A_acc = (accArrayMIM_A>1).sum() / numAttackSamples
    print('ALL_MIM_A_acc: ', ALL_MV_MIM_A_acc.data.cpu().numpy())
    ALL_MV_MIM_A_acc = (accArrayMIM_A==0).sum() / numAttackSamples
    print('MIM_A attack successful rate: ', ALL_MV_MIM_A_acc.data.cpu().numpy())
    torch.cuda.empty_cache()


    if ModelPlus_SNN_vgg.modelName=='SNN VGG-16 Backprop':
        advLoaderMIM = AttackWrappersWhiteBoxSNN.MIMNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, decayFactor, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    elif ModelPlus_SNN_vgg.modelName=='SNN ResNet Backprop':
        advLoaderMIM = AttackWrappersWhiteBoxJelly.MIMNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg,
                                                                  decayFactor, epsMax, epsStep, numSteps, clipMin,
                                                                  clipMax, targeted=False)
    else:
        advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, SNN_model_vgg,
                                                                     ModelPlus_SNN_vgg, decayFactor, epsMax, epsStep,
                                                                     numSteps, clipMin, clipMax, targeted=False)
    accArrayMIM_B = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderMIM)
        accArrayMIM_B = accArrayMIM_B + accArray
        print("MIM_B Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_MIM_B_acc = (accArrayMIM_B>=1).sum() / numAttackSamples
    print('MV_MIM_B_acc: ', MV_MIM_B_acc.data.cpu().numpy())
    ALL_MV_MIM_B_acc = (accArrayMIM_B>1).sum() / numAttackSamples
    print('ALL_MV_MIM_B_acc: ', ALL_MV_MIM_B_acc.data.cpu().numpy())
    ALL_MV_MIM_B_acc = (accArrayMIM_B==0).sum() / numAttackSamples
    print('MIM_B attack successful rate: ', ALL_MV_MIM_B_acc.data.cpu().numpy())
    torch.cuda.empty_cache()

    # advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, SNN_model_res, ModelPlus_SNN_res, decayFactor, epsMax, epsStep,
    #                                                         numSteps, clipMin, clipMax, targeted=False)
    # accArrayMIM_C = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(advLoaderMIM)
    #     accArrayMIM_C = accArrayMIM_C + accArray
    #     print("MIM_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    # MV_MIM_C_acc = (accArrayMIM_C>=1).sum() / numAttackSamples
    # print('MV_MIM_C_acc: ', MV_MIM_C_acc.data.cpu().numpy())
    # ALL_MV_MIM_C_acc = (accArrayMIM_C>1).sum() / numAttackSamples
    # print('ALL_MV_MIM_C_acc: ', ALL_MV_MIM_C_acc.data.cpu().numpy())
    # ALL_MV_MIM_C_acc = (accArrayMIM_C==0).sum() / numAttackSamples
    # print('MIM_C attack successful rate: ', ALL_MV_MIM_C_acc.data.cpu().numpy())
    torch.cuda.empty_cache()
    print()
    print('MV_MIM_acc: ', torch.min(MV_MIM_A_acc,MV_MIM_B_acc).data.cpu().numpy())
    # print('MV_MIM_acc: ', torch.min(torch.min(MV_MIM_A_acc,MV_MIM_B_acc), MV_MIM_C_acc).data.cpu().numpy())
    # print('ALL_MV_MIM_acc: ', torch.min(torch.min(ALL_MV_MIM_A_acc,ALL_MV_MIM_B_acc), ALL_MV_MIM_C_acc).data.cpu().numpy())
    print()

    print('PGD attack')
    if ModelPlus_CNN.modelName == 'SNN ResNet Backprop':
        advLoaderPGD = AttackWrappersWhiteBoxJelly.PGDNativePytorch(device, cleanLoader, syntheticModel,
                                                                    ModelPlus_CNN, epsMax, epsStep, numSteps,
                                                                    clipMin, clipMax, targeted=False)
    else:
        advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    accArrayPGD_A = torch.zeros(numAttackSamples).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_A = accArrayPGD_A + accArray
        print("PGD_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_PGD_A_acc = (accArrayPGD_A>=1).sum() / numAttackSamples
    print('MV_PGD_A_acc: ', MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A>1).sum() / numAttackSamples
    print('ALL_MV_PGD_A_acc: ', ALL_MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A==0).sum() / numAttackSamples
    print('PGD_A attack successful rate: ', ALL_MV_PGD_A_acc.data.cpu().numpy())

    if ModelPlus_SNN_vgg.modelName=='SNN VGG-16 Backprop':
        advLoaderPGD = AttackWrappersWhiteBoxSNN.PGDNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    elif ModelPlus_SNN_vgg.modelName=='SNN ResNet Backprop':
        advLoaderPGD = AttackWrappersWhiteBoxJelly.PGDNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    else:
        advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    accArrayPGD_B = torch.zeros(numAttackSamples).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_B = accArrayPGD_B + accArray
        print("PGD_B Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_PGD_B_acc = (accArrayPGD_B>=1).sum() / numAttackSamples
    print('MV_PGD_B_acc: ', MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B>1).sum() / numAttackSamples
    print('ALL_MV_PGD_B_acc: ', ALL_MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B==0).sum() / numAttackSamples
    print('PGD_B attack successful rate: ', ALL_MV_PGD_B_acc.data.cpu().numpy())

    # advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, SNN_model_res, ModelPlus_SNN_res, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    # accArrayPGD_C = torch.zeros(numAttackSamples).to(device)
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(advLoaderPGD)
    #     accArrayPGD_C = accArrayPGD_C + accArray
    #     print("PGD_C Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    # MV_PGD_C_Acc = (accArrayPGD_C>=2).sum() / numAttackSamples
    # print('MV_PGD_C_Acc: ', MV_PGD_C_Acc.data.cpu().numpy())
    # ALL_MV_PGD_C_Acc = (accArrayPGD_C>2).sum() / numAttackSamples
    # print('ALL_MV_PGD_C_Acc: ', ALL_MV_PGD_C_Acc.data.cpu().numpy())
    # ALL_MV_PGD_C_Acc = (accArrayPGD_C==0).sum() / numAttackSamples
    # print('PGD_C attack successful rate: ', ALL_MV_PGD_C_Acc.data.cpu().numpy())

    print()
    print('MV_PGD_acc: ', torch.min(MV_PGD_A_acc,MV_PGD_B_acc).data.cpu().numpy())
    # print('MV_PGD_acc: ', torch.min(torch.min(MV_PGD_A_acc,MV_PGD_B_acc), MV_PGD_C_Acc).data.cpu().numpy())
    # print('ALL_MV_PGD_acc: ', torch.min(torch.min(ALL_MV_PGD_A_acc,ALL_MV_PGD_B_acc), ALL_MV_PGD_C_Acc).data.cpu().numpy())
    print()
    # #Go through and check the robust accuray of each model on the adversarial examples
    # for i in range(0, len(modelPlusList)):
    #     acc = modelPlusList[i].validateD(advLoader)
    #     print(modelPlusList[i].modelName+" Robust Acc:", acc)


def SNN_AutoSAGA_two_bp(modelDir1, modelDir2, dataset):
    saveTag ="AutoSAGA Attack SNN"
    device = torch.device("cuda")
    # dataset = 'CIFAR10'
    imgSize = 32
    # batchSize = 64
    batchSize = 32
    if dataset == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        # mean = (0.5)
        # std = (0.5)
        numClasses = 10
        valLoader = DMP.GetCIFAR10Validation_norm(imgSize, batchSize, mean=0, std=1)

    elif dataset == 'CIFAR100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        numClasses = 100
        valLoader = DMP.GetCIFAR100Validation_norm(imgSize, batchSize, mean=0, std=1)
    #Load the defense
    # defense = Loadresnet(modelDir)
    modelPlusList = []

    #Create the synthetic model
    # syntheticModel = NetworkConstructorsAdaptive.CarliniNetwork(imgSize, numClasses)
    # # ================================Load CNN models==============================================
    # # syntheticModel = resnet56.resnet56(32, numClasses)
    # syntheticModel = VGG(vgg_name='VGG16', labels=numClasses, mean=mean, std=std)
    # # syntheticModel = ResNet20(labels=numClasses, dropout=0.2)
    # # syntheticModel = nn.DataParallel(syntheticModel)
    # checkpoint = torch.load(modelDir2)
    # syntheticModel.load_state_dict(checkpoint['state_dict'])
    # syntheticModel.to(device)
    # syntheticModel.eval()
    # ModelPlus_CNN = ModelPlus.ModelPlus("VGG16", syntheticModel, device, imgSize, imgSize, batchSize)
    # modelPlusList.append(ModelPlus_CNN)
    # synAcc = DMP.validateD(valLoader, syntheticModel)
    # print('load CNN from: ', modelDir2)
    # print("CNN Accuracy: ", synAcc)
    # # ================================Load CNN models==============================================

    valLoader_resized = None
    # # ================================Load ViT models==============================================
    # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadBiT_R101()
    syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadBiT_R50()
    # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadViTL()
    # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadViTB()
    
    syntheticModel.to(device)
    syntheticModel.eval()
    modelPlusList.append(ModelPlus_CNN)
    synAcc = DMP.validateD(valLoader_resized, syntheticModel)
    print("CNN Accuracy: ", synAcc)
    # # ================================Load ViT models==============================================

    # ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN(modelDir1, dataset, batchSize, mean, std)
    # ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN_resnet(modelDir1, dataset, batchSize, mean, std)
    SNN_model_vgg, ModelPlus_SNN_vgg, valLoader_repeated = LoadVGG16SNN_bp(batchSize=64)

    # ====================load RESNET BP SNN=================================================
    # SNN_model_vgg = SpikingJellyResNet.LoadCIFAR10SNNResNetBackProp("./resnet18acc_8183.pt")
    SNN_model_vgg = SpikingJellyResNet.LoadCIFAR100SNNResNetBackProp("./resnet18_acc6503_ver1.pt")
    # print('load snn resent bp ./resnet18acc_8183.pt' )
    print('load snn resent bp ./resnet18_acc6503_ver1.pt' )
    SNN_model_vgg.eval()
    ModelPlus_SNN_vgg = ModelPlus.ModelPlusSpikingJelly("SNN ResNet Backprop", SNN_model_vgg, device, 128, 128, batchSize)
    # ====================load RESNET BP SNN=================================================

    modelPlusList.append(ModelPlus_SNN_vgg)
    SNN_model_vgg.to(device)
    # transvggAcc = DMP.validateD(valLoader, SNN_model_vgg)
    transvggAcc = ModelPlus_SNN_vgg.validateD(valLoader)
    print("trans snn Accuracy: ", transvggAcc)
    # transvggAcc = ModelPlus_SNN_vgg.validateD(valLoader_repeated)
    # print("trans snn Accuracy: ", transvggAcc)

    # defense_res, ModelPlus_SNN_res, SNN_model_res = LoadDietSNN_resnet(modelDir1, dataset, batchSize, mean, std)
    # modelPlusList.append(ModelPlus_SNN_res)
    # SNN_model_res.to(device)
    # transresAcc = DMP.validateD(valLoader, SNN_model_res)
    # print("trans resnet Accuracy: ", transresAcc)

    # xTest, yTest = DMP.DataLoaderToTensor(valLoader_repeated)
    numModels = 2
    totalSampleNum = len(valLoader.dataset)
    # Get accuracy array for each model
    accArrayCumulative = torch.zeros(totalSampleNum).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(valLoader)
        accArrayCumulative = accArrayCumulative + accArray
    MV_clean_acc = (accArrayCumulative==2).sum() / totalSampleNum
    print('All_clean_acc: ', MV_clean_acc.data.cpu().numpy())
    MV_clean_acc = (accArrayCumulative>0).sum() / totalSampleNum
    print('clean_acc: ', MV_clean_acc.data.cpu().numpy())

    #Attack parameters
    numAttackSamples = 1000
    print('numAttackSamples: ', numAttackSamples)

    if valLoader_resized is None:
        cleanLoader = AttackWrappersProtoSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader,
                                                                  modelPlusList)
    else:
        cleanLoader = AttackWrappersProtoSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader_resized,
                                                                  modelPlusList)

    # (cleanData, cleanTarget) = torch.load('./Clean_Mon Aug 15 17:47:23 2022.pt')
    # cleanLoader = DMP.TensorToDataLoader(cleanData, cleanTarget, transforms=None, batchSize=10,
    #                                            randomizer=None)

    #Do the attack
    # dataLoaderForTraining = trainLoader
    # epsForAttacks = 0.05
    epsMax = 0.031
    # epsForAttacks = 0.062
    clipMin = 0.0
    clipMax = 1.0
    # clipMin = -1.0
    # clipMax = 1.0
    decayFactor = 1.0
    numSteps = 15
    # numSteps = 10
    # numSteps = 7
    # epsStep = epsForAttacks/numSteps
    epsStep = 0.005
    # epsStep = 0.01
    print('numSteps: ', numSteps, 'epsStep: ',epsStep, 'epsMax: ', epsMax)

    alphaLearningRate = 10000#0#100000
    # alphaLearningRate = 10000#0#100000
    # alphaLearningRate = 10000#0#100000
    fittingFactor = 50.0
    print("Alpha Learning Rate:", alphaLearningRate)
    print("Fitting Factor:", fittingFactor)

    torch.cuda.empty_cache()

    (advData, advTarget) = torch.load('./AutoSAGA_Mon Aug 15 17:47:23 2022.pt')
    advLoader = DMP.TensorToDataLoader(advData, advTarget, transforms=None, batchSize=10,
                                               randomizer=None)
    advLoader = AttackWrappersProtoSAGA.SelfAttentionGradientAttackProto(device, epsMax, epsStep, numSteps, modelPlusList,
                                                    cleanLoader, clipMin, clipMax, alphaLearningRate, fittingFactor, advLoader, numClasses=numClasses)
    accArrayProtoSAGA = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoader)
        accArrayProtoSAGA = accArrayProtoSAGA + accArray
        print("ProtoSAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_ProtoSAGA_acc = (accArrayProtoSAGA>=1).sum() / numAttackSamples
    print('MV_ProtoSAGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    ALL_MV_ProtoSAGA_acc = (accArrayProtoSAGA==2).sum() / numAttackSamples
    print('ALL_MV_ProtoSAGA_acc: ', ALL_MV_ProtoSAGA_acc.data.cpu().numpy())
    MV_ProtoSAGA_acc = (accArrayProtoSAGA==0).sum() / numAttackSamples
    print('ProtoSAGA attack successful rate: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    # torch.save(advLoaderMIM, saveDir+"//AdvLoaderMIM")
    torch.cuda.empty_cache()


    print()
    print('Basic SAGA attack')
    coefficientArray = torch.zeros(2)
    # secondcoeff = 2.0000e-04
    secondcoeff = 0.5
    coefficientArray[0] = 1.0 - secondcoeff
    coefficientArray[1] = secondcoeff
    print("Coeff Array:")
    print(coefficientArray)
    advLoader = AttackWrappersWhiteBoxP_SAGA.SelfAttentionGradientAttack(device, epsMax, numSteps, epsStep,
                                                modelPlusList, coefficientArray, cleanLoader, clipMin, clipMax, mean, std)
    accArraySAGA = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoader)
        accArraySAGA = accArraySAGA + accArray
        print("SAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_ProtoSAGA_acc = (accArraySAGA>=1).sum() / numAttackSamples
    print('MV_AGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    ALL_MV_SAGA_acc = (accArraySAGA>1).sum() / numAttackSamples
    print('ALL_MV_SAGA_acc: ', ALL_MV_SAGA_acc.data.cpu().numpy())
    MV_SAGA_acc = (accArraySAGA==0).sum() / numAttackSamples
    print('SAGA attack successful rate: ', MV_SAGA_acc.data.cpu().numpy())


    print()
    numSteps = 25
    # numSteps = 10
    # numSteps = 7
    # epsStep = epsForAttacks/numSteps
    # epsStep = 0.005
    epsStep = 0.01
    print('numSteps: ', numSteps, 'epsStep: ',epsStep, 'epsMax: ', epsMax)
    print('MIM attack')
    advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN, decayFactor, epsMax, epsStep,
                                                            numSteps, clipMin, clipMax, targeted=False)
    accArrayMIM_A = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderMIM)
        accArrayMIM_A = accArrayMIM_A + accArray
        print("MIM_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_MIM_A_acc = (accArrayMIM_A>=1).sum() / numAttackSamples
    print('MIM_A_acc: ', MV_MIM_A_acc.data.cpu().numpy())
    ALL_MV_MIM_A_acc = (accArrayMIM_A>1).sum() / numAttackSamples
    print('ALL_MIM_A_acc: ', ALL_MV_MIM_A_acc.data.cpu().numpy())
    ALL_MV_MIM_A_acc = (accArrayMIM_A==0).sum() / numAttackSamples
    print('MIM_A attack successful rate: ', ALL_MV_MIM_A_acc.data.cpu().numpy())
    torch.cuda.empty_cache()


    if ModelPlus_SNN_vgg.modelName=='SNN VGG-16 Backprop':
        advLoaderMIM = AttackWrappersWhiteBoxSNN.MIMNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, decayFactor, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    elif ModelPlus_SNN_vgg.modelName=='SNN ResNet Backprop':
        advLoaderMIM = AttackWrappersWhiteBoxJelly.MIMNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg,
                                                                  decayFactor, epsMax, epsStep, numSteps, clipMin,
                                                                  clipMax, targeted=False)
    else:
        advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, SNN_model_vgg,
                                                                     ModelPlus_SNN_vgg, decayFactor, epsMax, epsStep,
                                                                     numSteps, clipMin, clipMax, targeted=False)
    accArrayMIM_B = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderMIM)
        accArrayMIM_B = accArrayMIM_B + accArray
        print("MIM_B Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_MIM_B_acc = (accArrayMIM_B>=1).sum() / numAttackSamples
    print('MV_MIM_B_acc: ', MV_MIM_B_acc.data.cpu().numpy())
    ALL_MV_MIM_B_acc = (accArrayMIM_B>1).sum() / numAttackSamples
    print('ALL_MV_MIM_B_acc: ', ALL_MV_MIM_B_acc.data.cpu().numpy())
    ALL_MV_MIM_B_acc = (accArrayMIM_B==0).sum() / numAttackSamples
    print('MIM_B attack successful rate: ', ALL_MV_MIM_B_acc.data.cpu().numpy())
    torch.cuda.empty_cache()

    # advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, SNN_model_res, ModelPlus_SNN_res, decayFactor, epsMax, epsStep,
    #                                                         numSteps, clipMin, clipMax, targeted=False)
    # accArrayMIM_C = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(advLoaderMIM)
    #     accArrayMIM_C = accArrayMIM_C + accArray
    #     print("MIM_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    # MV_MIM_C_acc = (accArrayMIM_C>=1).sum() / numAttackSamples
    # print('MV_MIM_C_acc: ', MV_MIM_C_acc.data.cpu().numpy())
    # ALL_MV_MIM_C_acc = (accArrayMIM_C>1).sum() / numAttackSamples
    # print('ALL_MV_MIM_C_acc: ', ALL_MV_MIM_C_acc.data.cpu().numpy())
    # ALL_MV_MIM_C_acc = (accArrayMIM_C==0).sum() / numAttackSamples
    # print('MIM_C attack successful rate: ', ALL_MV_MIM_C_acc.data.cpu().numpy())
    torch.cuda.empty_cache()
    print()
    print('MV_MIM_acc: ', torch.min(MV_MIM_A_acc,MV_MIM_B_acc).data.cpu().numpy())
    # print('MV_MIM_acc: ', torch.min(torch.min(MV_MIM_A_acc,MV_MIM_B_acc), MV_MIM_C_acc).data.cpu().numpy())
    # print('ALL_MV_MIM_acc: ', torch.min(torch.min(ALL_MV_MIM_A_acc,ALL_MV_MIM_B_acc), ALL_MV_MIM_C_acc).data.cpu().numpy())
    print()

    print('PGD attack')
    advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    accArrayPGD_A = torch.zeros(numAttackSamples).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_A = accArrayPGD_A + accArray
        print("PGD_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_PGD_A_acc = (accArrayPGD_A>=1).sum() / numAttackSamples
    print('MV_PGD_A_acc: ', MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A>1).sum() / numAttackSamples
    print('ALL_MV_PGD_A_acc: ', ALL_MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A==0).sum() / numAttackSamples
    print('PGD_A attack successful rate: ', ALL_MV_PGD_A_acc.data.cpu().numpy())

    if ModelPlus_SNN_vgg.modelName=='SNN VGG-16 Backprop':
        advLoaderPGD = AttackWrappersWhiteBoxSNN.PGDNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    elif ModelPlus_SNN_vgg.modelName=='SNN ResNet Backprop':
        advLoaderPGD = AttackWrappersWhiteBoxJelly.PGDNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    else:
        advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    accArrayPGD_B = torch.zeros(numAttackSamples).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_B = accArrayPGD_B + accArray
        print("PGD_B Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_PGD_B_acc = (accArrayPGD_B>=1).sum() / numAttackSamples
    print('MV_PGD_B_acc: ', MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B>1).sum() / numAttackSamples
    print('ALL_MV_PGD_B_acc: ', ALL_MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B==0).sum() / numAttackSamples
    print('PGD_B attack successful rate: ', ALL_MV_PGD_B_acc.data.cpu().numpy())

    # advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, SNN_model_res, ModelPlus_SNN_res, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    # accArrayPGD_C = torch.zeros(numAttackSamples).to(device)
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(advLoaderPGD)
    #     accArrayPGD_C = accArrayPGD_C + accArray
    #     print("PGD_C Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    # MV_PGD_C_Acc = (accArrayPGD_C>=2).sum() / numAttackSamples
    # print('MV_PGD_C_Acc: ', MV_PGD_C_Acc.data.cpu().numpy())
    # ALL_MV_PGD_C_Acc = (accArrayPGD_C>2).sum() / numAttackSamples
    # print('ALL_MV_PGD_C_Acc: ', ALL_MV_PGD_C_Acc.data.cpu().numpy())
    # ALL_MV_PGD_C_Acc = (accArrayPGD_C==0).sum() / numAttackSamples
    # print('PGD_C attack successful rate: ', ALL_MV_PGD_C_Acc.data.cpu().numpy())

    print()
    print('MV_PGD_acc: ', torch.min(MV_PGD_A_acc,MV_PGD_B_acc).data.cpu().numpy())
    # print('MV_PGD_acc: ', torch.min(torch.min(MV_PGD_A_acc,MV_PGD_B_acc), MV_PGD_C_Acc).data.cpu().numpy())
    # print('ALL_MV_PGD_acc: ', torch.min(torch.min(ALL_MV_PGD_A_acc,ALL_MV_PGD_B_acc), ALL_MV_PGD_C_Acc).data.cpu().numpy())
    print()
    # #Go through and check the robust accuray of each model on the adversarial examples
    # for i in range(0, len(modelPlusList)):
    #     acc = modelPlusList[i].validateD(advLoader)
    #     print(modelPlusList[i].modelName+" Robust Acc:", acc)


def SNN_AutoSAGA_imagenet_two(modelDir1, modelDir2, dataset, imgSize=224):
    saveTag ="AutoSAGA Attack SNN"
    device = torch.device("cuda")
    # dataset = 'CIFAR10'

    batchSize = 5
    batchSize512 = 1
    # batchSize = 2
    # batchSize = 128
    # batchSize = 64
    if dataset == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        imgSize = 32
        # mean = (0.5)
        # std = (0.5)
        numClasses = 10
        valLoader = DMP.GetCIFAR10Validation_norm(imgSize, batchSize, mean=0, std=1)

    elif dataset == 'CIFAR100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        numClasses = 100
        imgSize = 32
        valLoader = DMP.GetCIFAR100Validation_norm(imgSize, batchSize, mean=0, std=1)
    elif dataset == 'IMAGENET':
        mean = 0
        std = 1
        numClasses = 1000
        # imgSize = 224
        valLoader = DMP.GetIMAGENETValidation(imgSize, batchSize)
    #Load the defense
    # defense = Loadresnet(modelDir)
    modelPlusList = []

    #Create the synthetic model
    # syntheticModel = NetworkConstructorsAdaptive.CarliniNetwork(imgSize, numClasses)
    # ================================Load CNN models==============================================
    # syntheticModel = resnet56.resnet56(32, numClasses)
    # syntheticModel = VGG(vgg_name='VGG16', labels=numClasses, mean=mean, std=std)
    #ResNet-152
    # syntheticModel = ResNetPretrainedImageNet.resnet152(pretrained=True)
    syntheticModel = VGG_imagenet.vgg16(pretrained=True)
    # syntheticModel = ResNet20(labels=numClasses, dropout=0.2)
    # syntheticModel = nn.DataParallel(syntheticModel)
    # checkpoint = torch.load(modelDir2)
    # syntheticModel.load_state_dict(checkpoint['state_dict'])
    syntheticModel.to(device)
    syntheticModel.eval()
    ModelPlus_CNN = ModelPlus.ModelPlus("VGG16", syntheticModel, device, imgSize, imgSize, batchSize)
    # ModelPlus_CNN = ModelPlus.ModelPlus("ResNet-152", syntheticModel, device, imgSize, imgSize, batchSize)
    modelPlusList.append(ModelPlus_CNN)
    # synAcc = DMP.validateD(valLoader, syntheticModel)
    # print("CNN Accuracy: ", synAcc)
    # ================================Load CNN models==============================================
    #
    # valLoader_resized = None
    # syntheticModel, ModelPlus_CNN, defense_CNN = LoadBiT_R152(device, batchSize512)
    #
    # # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadViTL_imagenet(device, imgSize)
    # # defense_CNN, ModelPlus_CNN, syntheticModel = LoadDietSNN(modelDir2, dataset, batchSize, mean, std, timesteps=20)
    # # syntheticModel.to(device)
    # syntheticModel.eval()
    # modelPlusList.append(ModelPlus_CNN)
    # # synAcc = DMP.validateD(valLoader, syntheticModel)
    # # # synAcc = DMP.validateD(valLoader_resized, syntheticModel)
    # # print("CNN Accuracy: ", synAcc)
    # # ================================Load model 1==============================================

    # ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN(modelDir1, dataset, batchSize, mean, std, timesteps=5)
    ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN_imgnet(modelDir1, batchSize, imgSize,device, timesteps=5)
    # ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN_resnet(modelDir1, dataset, batchSize, mean, std)
    modelPlusList.append(ModelPlus_SNN_vgg)
    # SNN_model_vgg.to(device)
    # transvggAcc = ModelPlus_SNN_vgg.validateD(valLoader)
    # print("trans snn Accuracy: ", transvggAcc)

    # defense_res, ModelPlus_SNN_res, SNN_model_res = LoadDietSNN_resnet(modelDir1, dataset, batchSize, mean, std)
    # modelPlusList.append(ModelPlus_SNN_res)
    # SNN_model_res.to(device)
    # transresAcc = DMP.validateD(valLoader, SNN_model_res)
    # print("trans resnet Accuracy: ", transresAcc)

    numModels = 2
    totalSampleNum = len(valLoader.dataset)
    # # Get accuracy array for each model
    # accArrayCumulative = torch.zeros(totalSampleNum).to(device)  # Create an array with one entry for ever sample in the dataset
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(valLoader)
    #     accArrayCumulative = accArrayCumulative + accArray
    # MV_clean_acc = (accArrayCumulative==2).sum() / totalSampleNum
    # print('All_clean_acc: ', MV_clean_acc.data.cpu().numpy())
    # MV_clean_acc = (accArrayCumulative>0).sum() / totalSampleNum
    # print('clean_acc: ', MV_clean_acc.data.cpu().numpy())

    for i in range(0, numModels):
        path = "./imgnet_accArray_imgnet_" + modelPlusList[i].modelName
        # if os.path.exists(path):
        accArray = torch.load(path)
        print("Clean Acc " + modelPlusList[i].modelName + ":", accArray.sum() / totalSampleNum)

    # sys.exit()

    #Attack parameters
    numAttackSamples = 1000
    print('numAttackSamples: ', numAttackSamples)
    cleanpath = "./imgnet_cleanLoader_imgnet_" + modelPlusList[0].modelName + "_" + modelPlusList[1].modelName
    if os.path.exists(cleanpath):
        cleanLoader = torch.load(cleanpath)
        (cleanData, cleanTarget) = DMP.DataLoaderToTensor(cleanLoader)
        if imgSize == 224:
            cleanLoader = DMP.TensorToDataLoader(cleanData, cleanTarget, transforms=None, batchSize=batchSize,
                                             randomizer=None)
        else:
            cleanLoader = DMP.TensorToDataLoader(cleanData, cleanTarget, transforms=None, batchSize=batchSize512,
                                                 randomizer=None)
    else:
        if imgSize == 224:
            cleanLoader = AttackWrappersProtoSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader,
                                                                      modelPlusList)
        else:
            valLoader224 = DMP.GetIMAGENETValidation(224, batchSize)
            cleanLoader = AttackWrappersProtoSAGA.GetFirstCorrectlyOverlappingSamplesBalanced_twosize(device, numAttackSamples, numClasses, valLoader, valLoader224,
                                                                      modelPlusList)
        torch.save(cleanLoader, "./imgnet_cleanLoader_imgnet_"+modelPlusList[0].modelName+"_"+modelPlusList[1].modelName)

    # cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, numAttackSamples, valLoader, numClasses)
    # SNNAcc_clean = DMP.validateD(cleanLoader, SNN_model)
    # print("SNN Model cleanLoader Accuracy: ", SNNAcc_clean)
    # (cleanData, cleanTarget) = torch.load('./Clean_Sun Aug 14 01:25:44 2022.pt')
    # cleanLoader = DMP.TensorToDataLoader(cleanData, cleanTarget, transforms=None, batchSize=10,
    #                                            randomizer=None)

    #Do the attack
    # dataLoaderForTraining = trainLoader
    # epsForAttacks = 0.05
    epsMax = 0.062
    # epsForAttacks = 0.062
    clipMin = 0.0
    clipMax = 1.0
    # clipMin = -1.0
    # clipMax = 1.0
    decayFactor = 1.0
    numSteps = 20
    # numSteps = 10
    # numSteps = 7
    # epsStep = epsForAttacks/numSteps
    epsStep = 0.005
    # epsStep = 0.01
    print('numSteps: ', numSteps, 'epsStep: ',epsStep, 'epsMax: ', epsMax)

    alphaLearningRate = 10000#0#100000
    # alphaLearningRate = 100000#0#100000
    # alphaLearningRate = 10000#0#100000
    fittingFactor = 50.0
    print("Alpha Learning Rate:", alphaLearningRate)
    print("Fitting Factor:", fittingFactor)

    torch.cuda.empty_cache()
    advLoader = AttackWrappersProtoSAGA.SelfAttentionGradientAttackProto(device, epsMax, epsStep, numSteps,
                                                                         modelPlusList,
                                                                        cleanLoader, clipMin, clipMax,
                                                                         alphaLearningRate, fittingFactor, advLoader=None,
                                                                         numClasses=numClasses)
    # (SAGAData, sagaTarget) = torch.load('./AutoSAGA_Tue Nov  1 04:45:27 2022.pt')
    # advLoader = DMP.TensorToDataLoader(SAGAData, sagaTarget, transforms=None, batchSize=batchSize,
    #                                            randomizer=None)
    accArrayProtoSAGA = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoader)
        accArrayProtoSAGA = accArrayProtoSAGA + accArray
        print("ProtoSAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_ProtoSAGA_acc = (accArrayProtoSAGA>=1).sum() / numAttackSamples
    print('MV_ProtoSAGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    ALL_MV_ProtoSAGA_acc = (accArrayProtoSAGA==2).sum() / numAttackSamples
    print('ALL_MV_ProtoSAGA_acc: ', ALL_MV_ProtoSAGA_acc.data.cpu().numpy())
    MV_ProtoSAGA_acc = (accArrayProtoSAGA==0).sum() / numAttackSamples
    print('ProtoSAGA attack successful rate: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    # torch.save(advLoaderMIM, saveDir+"//AdvLoaderMIM")
    torch.cuda.empty_cache()

    # sys.exit()

    print('Basic SAGA attack')
    coefficientArray = torch.zeros(2)
    # secondcoeff = 2.0000e-04
    secondcoeff = 0.5
    coefficientArray[0] = 1.0 - secondcoeff
    coefficientArray[1] = secondcoeff
    print("Coeff Array:")
    print(coefficientArray)
    advLoader = AttackWrappersWhiteBoxP_SAGA.SelfAttentionGradientAttack(device, epsMax, numSteps, epsStep,
                                                modelPlusList, coefficientArray, cleanLoader, clipMin, clipMax, mean, std)

    accArraySAGA = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoader)
        accArraySAGA = accArraySAGA + accArray
        print("SAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_ProtoSAGA_acc = (accArraySAGA>=1).sum() / numAttackSamples
    print('MV_AGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    ALL_MV_SAGA_acc = (accArraySAGA>1).sum() / numAttackSamples
    print('ALL_MV_SAGA_acc: ', ALL_MV_SAGA_acc.data.cpu().numpy())
    MV_SAGA_acc = (accArraySAGA==0).sum() / numAttackSamples
    print('SAGA attack successful rate: ', MV_SAGA_acc.data.cpu().numpy())

    print()
    numSteps = 25
    epsStep = 0.01
    print('numSteps: ', numSteps, 'epsStep: ', epsStep, 'epsMax: ', epsMax)
    print()
    print('MIM attack')
    advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN, decayFactor, epsMax, epsStep,
                                                            numSteps, clipMin, clipMax, targeted=False)
    accArrayMIM_A = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderMIM)
        accArrayMIM_A = accArrayMIM_A + accArray
        print("MIM_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_MIM_A_acc = (accArrayMIM_A>=1).sum() / numAttackSamples
    print('MIM_A_acc: ', MV_MIM_A_acc.data.cpu().numpy())
    ALL_MV_MIM_A_acc = (accArrayMIM_A>1).sum() / numAttackSamples
    print('ALL_MIM_A_acc: ', ALL_MV_MIM_A_acc.data.cpu().numpy())
    ALL_MV_MIM_A_acc = (accArrayMIM_A==0).sum() / numAttackSamples
    print('MIM_A attack successful rate: ', ALL_MV_MIM_A_acc.data.cpu().numpy())
    torch.cuda.empty_cache()

    advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, decayFactor, epsMax, epsStep,
                                                            numSteps, clipMin, clipMax, targeted=False)
    accArrayMIM_B = torch.zeros(numAttackSamples).to(device).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderMIM)
        accArrayMIM_B = accArrayMIM_B + accArray
        print("MIM_B Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_MIM_B_acc = (accArrayMIM_B>=1).sum() / numAttackSamples
    print('MV_MIM_B_acc: ', MV_MIM_B_acc.data.cpu().numpy())
    ALL_MV_MIM_B_acc = (accArrayMIM_B>1).sum() / numAttackSamples
    print('ALL_MV_MIM_B_acc: ', ALL_MV_MIM_B_acc.data.cpu().numpy())
    ALL_MV_MIM_B_acc = (accArrayMIM_B==0).sum() / numAttackSamples
    print('MIM_B attack successful rate: ', ALL_MV_MIM_B_acc.data.cpu().numpy())
    torch.cuda.empty_cache()

    # advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, SNN_model_res, ModelPlus_SNN_res, decayFactor, epsMax, epsStep,
    #                                                         numSteps, clipMin, clipMax, targeted=False)
    # accArrayMIM_C = torch.zeros(numAttackSamples).to(device).to(device)  # Create an array with one entry for ever sample in the dataset
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(advLoaderMIM)
    #     accArrayMIM_C = accArrayMIM_C + accArray
    #     print("MIM_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    # MV_MIM_C_acc = (accArrayMIM_C>=1).sum() / numAttackSamples
    # print('MV_MIM_C_acc: ', MV_MIM_C_acc.data.cpu().numpy())
    # ALL_MV_MIM_C_acc = (accArrayMIM_C>1).sum() / numAttackSamples
    # print('ALL_MV_MIM_C_acc: ', ALL_MV_MIM_C_acc.data.cpu().numpy())
    # ALL_MV_MIM_C_acc = (accArrayMIM_C==0).sum() / numAttackSamples
    # print('MIM_C attack successful rate: ', ALL_MV_MIM_C_acc.data.cpu().numpy())
    torch.cuda.empty_cache()
    print()
    print('MV_MIM_acc: ', torch.min(MV_MIM_A_acc,MV_MIM_B_acc).data.cpu().numpy())
    # print('MV_MIM_acc: ', torch.min(torch.min(MV_MIM_A_acc,MV_MIM_B_acc), MV_MIM_C_acc).data.cpu().numpy())
    # print('ALL_MV_MIM_acc: ', torch.min(torch.min(ALL_MV_MIM_A_acc,ALL_MV_MIM_B_acc), ALL_MV_MIM_C_acc).data.cpu().numpy())
    print()

    print('PGD attack')
    advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    accArrayPGD_A = torch.zeros(numAttackSamples).to(device).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_A = accArrayPGD_A + accArray
        print("PGD_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_PGD_A_acc = (accArrayPGD_A>=1).sum() / numAttackSamples
    print('MV_PGD_A_acc: ', MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A>1).sum() / numAttackSamples
    print('ALL_MV_PGD_A_acc: ', ALL_MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A==0).sum() / numAttackSamples
    print('PGD_A attack successful rate: ', ALL_MV_PGD_A_acc.data.cpu().numpy())

    advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    accArrayPGD_B = torch.zeros(numAttackSamples).to(device).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_B = accArrayPGD_B + accArray
        print("PGD_B Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_PGD_B_acc = (accArrayPGD_B>=1).sum() / numAttackSamples
    print('MV_PGD_B_acc: ', MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B>1).sum() / numAttackSamples
    print('ALL_MV_PGD_B_acc: ', ALL_MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B==0).sum() / numAttackSamples
    print('PGD_B attack successful rate: ', ALL_MV_PGD_B_acc.data.cpu().numpy())

    # advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, SNN_model_res, ModelPlus_SNN_res, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    # accArrayPGD_C = torch.zeros(numAttackSamples).to(device).to(device)
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(advLoaderPGD)
    #     accArrayPGD_C = accArrayPGD_C + accArray
    #     print("PGD_C Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    # MV_PGD_C_Acc = (accArrayPGD_C>=2).sum() / numAttackSamples
    # print('MV_PGD_C_Acc: ', MV_PGD_C_Acc.data.cpu().numpy())
    # ALL_MV_PGD_C_Acc = (accArrayPGD_C>2).sum() / numAttackSamples
    # print('ALL_MV_PGD_C_Acc: ', ALL_MV_PGD_C_Acc.data.cpu().numpy())
    # ALL_MV_PGD_C_Acc = (accArrayPGD_C==0).sum() / numAttackSamples
    # print('PGD_C attack successful rate: ', ALL_MV_PGD_C_Acc.data.cpu().numpy())

    print()
    print('MV_PGD_acc: ', torch.min(MV_PGD_A_acc,MV_PGD_B_acc).data.cpu().numpy())
    # print('MV_PGD_acc: ', torch.min(torch.min(MV_PGD_A_acc,MV_PGD_B_acc), MV_PGD_C_Acc).data.cpu().numpy())
    # print('ALL_MV_PGD_acc: ', torch.min(torch.min(ALL_MV_PGD_A_acc,ALL_MV_PGD_B_acc), ALL_MV_PGD_C_Acc).data.cpu().numpy())
    print()
    # #Go through and check the robust accuray of each model on the adversarial examples
    # for i in range(0, len(modelPlusList)):
    #     acc = modelPlusList[i].validateD(advLoader)
    #     print(modelPlusList[i].modelName+" Robust Acc:", acc)


def SNN_AutoSAGA_imagenet_two_bp(modelDir1, modelDir2, dataset, imgSize = 224):
    saveTag = "AutoSAGA Attack imagenet SNN"
    device = torch.device("cuda")
    # dataset = 'CIFAR10'
    # imgSize = 32
    # batchSize = 256
    # batchSize = 128
    batchSize = 16
    # batchSize = 32
    batchSize512 = 1
    # batchSize = 2
    if dataset == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        # mean = (0.5)
        # std = (0.5)
        numClasses = 10
        valLoader = DMP.GetCIFAR10Validation_norm(imgSize, batchSize, mean=0, std=1)

    elif dataset == 'CIFAR100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        numClasses = 100
        valLoader = DMP.GetCIFAR100Validation_norm(imgSize, batchSize, mean=0, std=1)
    elif dataset == 'IMAGENET':
        mean = 0
        std = 1
        numClasses = 1000
        # imgSize = 224
        valLoader = DMP.GetIMAGENETValidation(imgSize, batchSize)
    # Load the defense
    # defense = Loadresnet(modelDir)
    modelPlusList = []

    # Create the synthetic model
    # syntheticModel = NetworkConstructorsAdaptive.CarliniNetwork(imgSize, numClasses)
    # # # ================================Load CNN models==============================================
    # # syntheticModel = resnet56.resnet56(32, numClasses)
    # # syntheticModel = VGG(vgg_name='VGG16', labels=numClasses, mean=mean, std=std)
    # #ResNet-152
    # # syntheticModel = ResNetPretrainedImageNet.resnet152(pretrained=True)
    # syntheticModel = VGG_imagenet.vgg16(pretrained=True)
    # # syntheticModel = ResNet20(labels=numClasses, dropout=0.2)
    # # syntheticModel = nn.DataParallel(syntheticModel)
    # # checkpoint = torch.load(modelDir2)
    # # syntheticModel.load_state_dict(checkpoint['state_dict'])
    # syntheticModel.to(device)
    # syntheticModel.eval()
    # ModelPlus_CNN = ModelPlus.ModelPlus("VGG16", syntheticModel, device, imgSize, imgSize, batchSize)
    # # ModelPlus_CNN = ModelPlus.ModelPlus("ResNet-152", syntheticModel, device, imgSize, imgSize, batchSize)
    # modelPlusList.append(ModelPlus_CNN)
    # # synAcc = DMP.validateD(valLoader, syntheticModel)
    # # print("CNN Accuracy: ", synAcc)
    # # # ================================Load CNN models==============================================

    valLoader_resized = None
    # ================================Load ViT models==============================================
    # syntheticModel, ModelPlus_CNN, defense_CNN = LoadBiT_R152(device, batchSize512)

    syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadViTL_imagenet(device, imgSize)

    # syntheticModel.to(device)
    # syntheticModel.eval()
    modelPlusList.append(ModelPlus_CNN)
    # synAcc = DMP.validateD(valLoader_resized, syntheticModel)
    # print("CNN Accuracy: ", synAcc)
    # ================================Load ViT models==============================================

    # ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN(modelDir1, dataset, batchSize, mean, std)
    # ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN_resnet(modelDir1, dataset, batchSize, mean, std)
    # SNN_model_vgg, ModelPlus_SNN_vgg, valLoader_repeated = LoadVGG16SNN_bp(batchSize=64)

    # ====================load RESNET BP SNN=================================================
    # SNN_model_vgg = SpikingJellyResNet.LoadCIFAR10SNNResNetBackProp("./resnet18acc_8183.pt")
    # SNN_model_vgg = SpikingJellyResNet.LoadCIFAR100SNNResNetBackProp("./resnet18_acc6503_ver1.pt")
    # # print('load snn resent bp ./resnet18acc_8183.pt' )
    # print('load snn resent bp ./resnet18_acc6503_ver1.pt')
    # SNN_model_vgg.eval()
    # ModelPlus_SNN_vgg = ModelPlus.ModelPlusSpikingJelly("SNN ResNet Backprop", SNN_model_vgg, device, 128, 128,
    #                                                     batchSize)

    arch = 'sew_resnet18'
    SNN_model_vgg, ModelPlus_SNN_vgg = LoadsjSNNresnet(arch, device, batchSize)
    # ====================load RESNET BP SNN=================================================

    modelPlusList.append(ModelPlus_SNN_vgg)
    # SNN_model_vgg.to(device)

    # transvggAcc = DMP.validateD(valLoader, SNN_model_vgg)
    # transvggAcc = ModelPlus_SNN_vgg.validateD(valLoader)
    # print("trans snn Accuracy: ", transvggAcc)
    # transvggAcc = ModelPlus_SNN_vgg.validateD(valLoader_repeated)
    # print("trans snn Accuracy: ", transvggAcc)

    # defense_res, ModelPlus_SNN_res, SNN_model_res = LoadDietSNN_resnet(modelDir1, dataset, batchSize, mean, std)
    # modelPlusList.append(ModelPlus_SNN_res)
    # SNN_model_res.to(device)
    # transresAcc = DMP.validateD(valLoader, SNN_model_res)
    # print("trans resnet Accuracy: ", transresAcc)

    # xTest, yTest = DMP.DataLoaderToTensor(valLoader_repeated)
    numModels = 2
    totalSampleNum = len(valLoader.dataset)
    # # Get accuracy array for each model
    # accArrayCumulative = torch.zeros(totalSampleNum).to(device)  # Create an array with one entry for ever sample in the dataset
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(valLoader)
    #     accArrayCumulative = accArrayCumulative + accArray
    # MV_clean_acc = (accArrayCumulative == 2).sum() / totalSampleNum
    # print('All_clean_acc: ', MV_clean_acc.data.cpu().numpy())
    # MV_clean_acc = (accArrayCumulative > 0).sum() / totalSampleNum
    # print('clean_acc: ', MV_clean_acc.data.cpu().numpy())

    # Attack parameters
    numAttackSamples = 1000
    print('numAttackSamples: ', numAttackSamples)
    cleanpath = "./imgnet_cleanLoader_imgnet_" + modelPlusList[0].modelName + "_" + modelPlusList[1].modelName
    if os.path.exists(cleanpath):
        cleanLoader = torch.load(cleanpath)
        (cleanData, cleanTarget) = DMP.DataLoaderToTensor(cleanLoader)
        if imgSize == 224:
            cleanLoader = DMP.TensorToDataLoader(cleanData, cleanTarget, transforms=None, batchSize=batchSize,
                                                 randomizer=None)
        else:
            cleanLoader = DMP.TensorToDataLoader(cleanData, cleanTarget, transforms=None, batchSize=batchSize512,
                                                 randomizer=None)
    else:
        if imgSize == 224:
            cleanLoader = AttackWrappersProtoSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader,
                                                                      modelPlusList)
        else:
            valLoader224 = DMP.GetIMAGENETValidation(224, batchSize)
            cleanLoader = AttackWrappersProtoSAGA.GetFirstCorrectlyOverlappingSamplesBalanced_twosize(device, numAttackSamples, numClasses, valLoader, valLoader224,
                                                                      modelPlusList)
        torch.save(cleanLoader, "./imgnet_cleanLoader_imgnet_"+modelPlusList[0].modelName+"_"+modelPlusList[1].modelName)
    # cleanLoader = torch.load('./imgnet_cleanLoader_imgnet_vgg16_sjSNNresnet')
    # (cleanData, cleanTarget) = DMP.DataLoaderToTensor(cleanLoader)
    # cleanLoader = DMP.TensorToDataLoader(cleanData, cleanTarget, transforms=None, batchSize=64,
    #                                      randomizer=None)

    # Do the attack
    # dataLoaderForTraining = trainLoader
    # epsForAttacks = 0.05
    epsMax = 0.062
    # epsForAttacks = 0.062
    clipMin = 0.0
    clipMax = 1.0
    # clipMin = -1.0
    # clipMax = 1.0
    decayFactor = 1.0
    # numSteps = 20
    numSteps = 10
    # numSteps = 7
    # epsStep = epsForAttacks/numSteps
    # epsStep = 0.005
    epsStep = 0.01
    print('numSteps: ', numSteps, 'epsStep: ', epsStep, 'epsMax: ', epsMax)

    alphaLearningRate = 10000  # 0#100000
    # alphaLearningRate = 10000#0#100000
    # alphaLearningRate = 10000#0#100000
    fittingFactor = 50.0
    print("Alpha Learning Rate:", alphaLearningRate)
    print("Fitting Factor:", fittingFactor)

    torch.cuda.empty_cache()

    (advData, advTarget) = torch.load('./AutoSAGA_Tue Nov  1 16:40:42 2022.pt')
    advLoader = DMP.TensorToDataLoader(advData, advTarget, transforms=None, batchSize=batchSize,
                                       randomizer=None)
    advLoader = AttackWrappersProtoSAGA.SelfAttentionGradientAttackProto(device, epsMax, epsStep, numSteps,
                                                                         modelPlusList,
                                                                         cleanLoader, clipMin, clipMax,
                                                                         alphaLearningRate, fittingFactor, advLoader=advLoader,
                                                                         numClasses=numClasses)
    # (SAGAData, sagaTarget) = torch.load('./AutoSAGA_Tue Nov  1 08:05:22 2022.pt')
    # advLoader = DMP.TensorToDataLoader(SAGAData, sagaTarget, transforms=None, batchSize=batchSize,
    #                                            randomizer=None)
    accArrayProtoSAGA = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoader)
        accArrayProtoSAGA = accArrayProtoSAGA + accArray
        print("ProtoSAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum() / numAttackSamples)
    MV_ProtoSAGA_acc = (accArrayProtoSAGA >= 1).sum() / numAttackSamples
    print('MV_ProtoSAGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    ALL_MV_ProtoSAGA_acc = (accArrayProtoSAGA == 2).sum() / numAttackSamples
    print('ALL_MV_ProtoSAGA_acc: ', ALL_MV_ProtoSAGA_acc.data.cpu().numpy())
    MV_ProtoSAGA_acc = (accArrayProtoSAGA == 0).sum() / numAttackSamples
    print('ProtoSAGA attack successful rate: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    # torch.save(advLoaderMIM, saveDir+"//AdvLoaderMIM")
    torch.cuda.empty_cache()
    sys.exit()


    print()
    print('Basic SAGA attack')
    coefficientArray = torch.zeros(2)
    # secondcoeff = 2.0000e-04
    secondcoeff = 0.5
    coefficientArray[0] = 1.0 - secondcoeff
    coefficientArray[1] = secondcoeff
    print("Coeff Array:")
    print(coefficientArray)
    # advLoader = AttackWrappersWhiteBoxP_SAGA.SelfAttentionGradientAttack(device, epsMax, numSteps, epsStep,
    #                                                                      modelPlusList, coefficientArray, cleanLoader,
    #                                                                      clipMin, clipMax, mean, std)

    (SAGAData, sagaTarget) = torch.load('./SAGA_Tue Nov  1 21:20:48 2022.pt')
    advLoader = DMP.TensorToDataLoader(SAGAData, sagaTarget, transforms=None, batchSize=batchSize,
                                               randomizer=None)
    accArraySAGA = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoader)
        accArraySAGA = accArraySAGA + accArray
        print("SAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum() / numAttackSamples)
    MV_ProtoSAGA_acc = (accArraySAGA >= 1).sum() / numAttackSamples
    print('MV_AGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    ALL_MV_SAGA_acc = (accArraySAGA > 1).sum() / numAttackSamples
    print('ALL_MV_SAGA_acc: ', ALL_MV_SAGA_acc.data.cpu().numpy())
    MV_SAGA_acc = (accArraySAGA == 0).sum() / numAttackSamples
    print('SAGA attack successful rate: ', MV_SAGA_acc.data.cpu().numpy())

    print()
    numSteps = 25
    # numSteps = 10
    # numSteps = 7
    # epsStep = epsForAttacks/numSteps
    # epsStep = 0.005
    epsStep = 0.01
    print('numSteps: ', numSteps, 'epsStep: ', epsStep, 'epsMax: ', epsMax)
    print('MIM attack')
    advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN,
                                                                 decayFactor, epsMax, epsStep,
                                                                 numSteps, clipMin, clipMax, targeted=False)
    accArrayMIM_A = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderMIM)
        accArrayMIM_A = accArrayMIM_A + accArray
        print("MIM_A Acc " + modelPlusList[i].modelName + ":", accArray.sum() / numAttackSamples)
    MV_MIM_A_acc = (accArrayMIM_A >= 1).sum() / numAttackSamples
    print('MIM_A_acc: ', MV_MIM_A_acc.data.cpu().numpy())
    ALL_MV_MIM_A_acc = (accArrayMIM_A > 1).sum() / numAttackSamples
    print('ALL_MIM_A_acc: ', ALL_MV_MIM_A_acc.data.cpu().numpy())
    ALL_MV_MIM_A_acc = (accArrayMIM_A == 0).sum() / numAttackSamples
    print('MIM_A attack successful rate: ', ALL_MV_MIM_A_acc.data.cpu().numpy())
    torch.cuda.empty_cache()

    if ModelPlus_SNN_vgg.modelName == 'SNN VGG-16 Backprop':
        advLoaderMIM = AttackWrappersWhiteBoxSNN.MIMNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg,
                                                                  decayFactor, epsMax, epsStep, numSteps, clipMin,
                                                                  clipMax, targeted=False)
    elif ModelPlus_SNN_vgg.modelName == 'SNN ResNet Backprop':
        advLoaderMIM = AttackWrappersWhiteBoxJelly.MIMNativePytorch(device, cleanLoader, SNN_model_vgg,
                                                                    ModelPlus_SNN_vgg,
                                                                    decayFactor, epsMax, epsStep, numSteps, clipMin,
                                                                    clipMax, targeted=False)
    else:
        advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, SNN_model_vgg,
                                                                     ModelPlus_SNN_vgg, decayFactor, epsMax, epsStep,
                                                                     numSteps, clipMin, clipMax, targeted=False)
    accArrayMIM_B = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderMIM)
        accArrayMIM_B = accArrayMIM_B + accArray
        print("MIM_B Acc " + modelPlusList[i].modelName + ":", accArray.sum() / numAttackSamples)
    MV_MIM_B_acc = (accArrayMIM_B >= 1).sum() / numAttackSamples
    print('MV_MIM_B_acc: ', MV_MIM_B_acc.data.cpu().numpy())
    ALL_MV_MIM_B_acc = (accArrayMIM_B > 1).sum() / numAttackSamples
    print('ALL_MV_MIM_B_acc: ', ALL_MV_MIM_B_acc.data.cpu().numpy())
    ALL_MV_MIM_B_acc = (accArrayMIM_B == 0).sum() / numAttackSamples
    print('MIM_B attack successful rate: ', ALL_MV_MIM_B_acc.data.cpu().numpy())
    torch.cuda.empty_cache()

    # advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, SNN_model_res, ModelPlus_SNN_res, decayFactor, epsMax, epsStep,
    #                                                         numSteps, clipMin, clipMax, targeted=False)
    # accArrayMIM_C = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(advLoaderMIM)
    #     accArrayMIM_C = accArrayMIM_C + accArray
    #     print("MIM_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    # MV_MIM_C_acc = (accArrayMIM_C>=1).sum() / numAttackSamples
    # print('MV_MIM_C_acc: ', MV_MIM_C_acc.data.cpu().numpy())
    # ALL_MV_MIM_C_acc = (accArrayMIM_C>1).sum() / numAttackSamples
    # print('ALL_MV_MIM_C_acc: ', ALL_MV_MIM_C_acc.data.cpu().numpy())
    # ALL_MV_MIM_C_acc = (accArrayMIM_C==0).sum() / numAttackSamples
    # print('MIM_C attack successful rate: ', ALL_MV_MIM_C_acc.data.cpu().numpy())
    torch.cuda.empty_cache()
    print()
    print('MV_MIM_acc: ', torch.min(MV_MIM_A_acc, MV_MIM_B_acc).data.cpu().numpy())
    # print('MV_MIM_acc: ', torch.min(torch.min(MV_MIM_A_acc,MV_MIM_B_acc), MV_MIM_C_acc).data.cpu().numpy())
    # print('ALL_MV_MIM_acc: ', torch.min(torch.min(ALL_MV_MIM_A_acc,ALL_MV_MIM_B_acc), ALL_MV_MIM_C_acc).data.cpu().numpy())
    print()

    print('PGD attack')
    advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN,
                                                                 epsMax, epsStep, numSteps, clipMin, clipMax,
                                                                 targeted=False)
    accArrayPGD_A = torch.zeros(numAttackSamples).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_A = accArrayPGD_A + accArray
        print("PGD_A Acc " + modelPlusList[i].modelName + ":", accArray.sum() / numAttackSamples)
    MV_PGD_A_acc = (accArrayPGD_A >= 1).sum() / numAttackSamples
    print('MV_PGD_A_acc: ', MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A > 1).sum() / numAttackSamples
    print('ALL_MV_PGD_A_acc: ', ALL_MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A == 0).sum() / numAttackSamples
    print('PGD_A attack successful rate: ', ALL_MV_PGD_A_acc.data.cpu().numpy())

    if ModelPlus_SNN_vgg.modelName == 'SNN VGG-16 Backprop':
        advLoaderPGD = AttackWrappersWhiteBoxSNN.PGDNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg,
                                                                  epsMax, epsStep, numSteps, clipMin, clipMax,
                                                                  targeted=False)
    elif ModelPlus_SNN_vgg.modelName == 'SNN ResNet Backprop':
        advLoaderPGD = AttackWrappersWhiteBoxJelly.PGDNativePytorch(device, cleanLoader, SNN_model_vgg,
                                                                    ModelPlus_SNN_vgg, epsMax, epsStep, numSteps,
                                                                    clipMin, clipMax, targeted=False)
    else:
        advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, SNN_model_vgg,
                                                                     ModelPlus_SNN_vgg, epsMax, epsStep, numSteps,
                                                                     clipMin, clipMax, targeted=False)
    accArrayPGD_B = torch.zeros(numAttackSamples).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_B = accArrayPGD_B + accArray
        print("PGD_B Acc " + modelPlusList[i].modelName + ":", accArray.sum() / numAttackSamples)
    MV_PGD_B_acc = (accArrayPGD_B >= 1).sum() / numAttackSamples
    print('MV_PGD_B_acc: ', MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B > 1).sum() / numAttackSamples
    print('ALL_MV_PGD_B_acc: ', ALL_MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B == 0).sum() / numAttackSamples
    print('PGD_B attack successful rate: ', ALL_MV_PGD_B_acc.data.cpu().numpy())

    # advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, SNN_model_res, ModelPlus_SNN_res, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    # accArrayPGD_C = torch.zeros(numAttackSamples).to(device)
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(advLoaderPGD)
    #     accArrayPGD_C = accArrayPGD_C + accArray
    #     print("PGD_C Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    # MV_PGD_C_Acc = (accArrayPGD_C>=2).sum() / numAttackSamples
    # print('MV_PGD_C_Acc: ', MV_PGD_C_Acc.data.cpu().numpy())
    # ALL_MV_PGD_C_Acc = (accArrayPGD_C>2).sum() / numAttackSamples
    # print('ALL_MV_PGD_C_Acc: ', ALL_MV_PGD_C_Acc.data.cpu().numpy())
    # ALL_MV_PGD_C_Acc = (accArrayPGD_C==0).sum() / numAttackSamples
    # print('PGD_C attack successful rate: ', ALL_MV_PGD_C_Acc.data.cpu().numpy())

    print()
    print('MV_PGD_acc: ', torch.min(MV_PGD_A_acc, MV_PGD_B_acc).data.cpu().numpy())
    # print('MV_PGD_acc: ', torch.min(torch.min(MV_PGD_A_acc,MV_PGD_B_acc), MV_PGD_C_Acc).data.cpu().numpy())
    # print('ALL_MV_PGD_acc: ', torch.min(torch.min(ALL_MV_PGD_A_acc,ALL_MV_PGD_B_acc), ALL_MV_PGD_C_Acc).data.cpu().numpy())
    print()
    # #Go through and check the robust accuray of each model on the adversarial examples
    # for i in range(0, len(modelPlusList)):
    #     acc = modelPlusList[i].validateD(advLoader)
    #     print(modelPlusList[i].modelName+" Robust Acc:", acc)


def SNN_AutoSAGA_imagenet_two_snnsnn(modelDir1, modelDir2, dataset, imgSize = 224):
    saveTag ="AutoSAGA Attack imagenet SNN"
    device = torch.device("cuda")
    # dataset = 'CIFAR10'
    # imgSize = 32
    # batchSize = 100
    batchSize = 10
    if dataset == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        # mean = (0.5)
        # std = (0.5)
        numClasses = 10
        valLoader = DMP.GetCIFAR10Validation_norm(imgSize, batchSize, mean=0, std=1)

    elif dataset == 'CIFAR100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        numClasses = 100
        valLoader = DMP.GetCIFAR100Validation_norm(imgSize, batchSize, mean, std)
    elif dataset == 'IMAGENET':
        mean = 0
        std = 1
        numClasses = 1000
        # imgSize = 224
        valLoader = DMP.GetIMAGENETValidation(imgSize, batchSize)

    #Load the defense
    # defense = Loadresnet(modelDir)
    modelPlusList = []

    #Create the synthetic model
    # syntheticModel = NetworkConstructorsAdaptive.CarliniNetwork(imgSize, numClasses)
    # # ================================Load CNN models==============================================
    # # syntheticModel = resnet56.resnet56(32, numClasses)
    # syntheticModel = VGG(vgg_name='VGG16', labels=numClasses, mean=mean, std=std)
    # # syntheticModel = ResNet20(labels=numClasses, dropout=0.2)
    # # syntheticModel = nn.DataParallel(syntheticModel)
    # checkpoint = torch.load(modelDir2)
    # syntheticModel.load_state_dict(checkpoint['state_dict'])
    # syntheticModel.to(device)
    # syntheticModel.eval()
    # ModelPlus_CNN = ModelPlus.ModelPlus("VGG16", syntheticModel, device, imgSize, imgSize, batchSize)
    # modelPlusList.append(ModelPlus_CNN)
    # synAcc = DMP.validateD(valLoader, syntheticModel)
    # print('load CNN from: ', modelDir2)
    # print("CNN Accuracy: ", synAcc)
    # # ================================Load CNN models==============================================

    valLoader_resized = None
    # # ================================Load ViT models==============================================
    # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadBiT_R101()
    # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadBiT_R50()
    # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadViTL()
    # syntheticModel, ModelPlus_CNN, defense_CNN, valLoader_resized = LoadViTB()
    # syntheticModel.to(device)
    # syntheticModel.eval()
    # modelPlusList.append(ModelPlus_CNN)
    # synAcc = DMP.validateD(valLoader_resized, syntheticModel)
    # print("CNN Accuracy: ", synAcc)
    # # ================================Load ViT models==============================================
    ## ====================load RESNET BP SNN=================================================
    # syntheticModel = SpikingJellyResNet.LoadCIFAR10SNNResNetBackProp("./resnet18acc_8183.pt")
    # print('load snn resent bp ./resnet18acc_8183.pt' )
    # ModelPlus_CNN = ModelPlus.ModelPlusSpikingJelly("SNN ResNet Backprop", syntheticModel, device, 32, 32, batchSize)
    arch = 'sew_resnet18'
    syntheticModel, ModelPlus_CNN = LoadsjSNNresnet(arch, device, batchSize=32)
    syntheticModel.to(device)
    syntheticModel.eval()
    modelPlusList.append(ModelPlus_CNN)
    # synAcc = ModelPlus_CNN.validateD(valLoader)
    # print("SNN ResNet Backprop Accuracy: ", synAcc)

    ## ====================load RESNET BP SNN=================================================

    ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN_imgnet(modelDir1, batchSize, imgSize,device, timesteps=5)
    # ModelPlus_SNN_vgg, SNN_model_vgg = LoadDietSNN_resnet(modelDir1, dataset, batchSize, mean, std)
    # SNN_model_vgg, ModelPlus_SNN_vgg, valLoader_repeated = LoadVGG16SNN_bp(batchSize=32)

    # # ====================load RESNET BP SNN=================================================
    # SNN_model_vgg = SpikingJellyResNet.LoadCIFAR10SNNResNetBackProp("./resnet18acc_8183.pt")
    # print('load snn resent bp ./resnet18acc_8183.pt' )
    # SNN_model_vgg.eval()
    # ModelPlus_SNN_vgg = ModelPlus.ModelPlusSpikingJelly("SNN ResNet Backprop", SNN_model_vgg, device, 32, 32, batchSize)
    # # ====================load RESNET BP SNN=================================================

    modelPlusList.append(ModelPlus_SNN_vgg)
    SNN_model_vgg.to(device)
    # # transvggAcc = DMP.validateD(valLoader, SNN_model_vgg)
    # transvggAcc = ModelPlus_SNN_vgg.validateD(valLoader)
    # # transvggAcc = DMP.validateD(valLoader_repeated, SNN_model_vgg)
    # print("trans snn Accuracy: ", transvggAcc)

    # defense_res, ModelPlus_SNN_res, SNN_model_res = LoadDietSNN_resnet(modelDir1, dataset, batchSize, mean, std)
    # modelPlusList.append(ModelPlus_SNN_res)
    # SNN_model_res.to(device)
    # transresAcc = DMP.validateD(valLoader, SNN_model_res)
    # print("trans resnet Accuracy: ", transresAcc)

    # xTest, yTest = DMP.DataLoaderToTensor(valLoader_repeated)
    numModels = 2
    totalSampleNum = len(valLoader.dataset)
    # # Get accuracy array for each model
    # accArrayCumulative = torch.zeros(totalSampleNum).to(device)  # Create an array with one entry for ever sample in the dataset
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(valLoader)
    #     accArrayCumulative = accArrayCumulative + accArray
    # MV_clean_acc = (accArrayCumulative==2).sum() / totalSampleNum
    # print('All_clean_acc: ', MV_clean_acc.data.cpu().numpy())
    # MV_clean_acc = (accArrayCumulative>0).sum() / totalSampleNum
    # print('clean_acc: ', MV_clean_acc.data.cpu().numpy())

    for i in range(0, numModels):
        path = "./imgnet_accArray_imgnet_" + modelPlusList[i].modelName
        # if os.path.exists(path):
        accArray = torch.load(path)
        print("Clean Acc " + modelPlusList[i].modelName + ":", accArray.sum() / totalSampleNum)

    sys.exit()

    #Attack parameters
    numAttackSamples = 1000
    print('numAttackSamples: ', numAttackSamples)

    cleanpath = "./imgnet_cleanLoader_imgnet_" + modelPlusList[0].modelName + "_" + modelPlusList[1].modelName
    if os.path.exists(cleanpath):
        cleanLoader = torch.load(cleanpath)
        (cleanData, cleanTarget) = DMP.DataLoaderToTensor(cleanLoader)
        cleanLoader = DMP.TensorToDataLoader(cleanData, cleanTarget, transforms=None, batchSize=batchSize,
                                             randomizer=None)
    else:
        if imgSize == 224:
            cleanLoader = AttackWrappersProtoSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, numAttackSamples, numClasses, valLoader,
                                                                      modelPlusList)
        else:
            valLoader224 = DMP.GetIMAGENETValidation(224, batchSize)
            cleanLoader = AttackWrappersProtoSAGA.GetFirstCorrectlyOverlappingSamplesBalanced_twosize(device, numAttackSamples, numClasses, valLoader, valLoader224,
                                                                      modelPlusList)
        torch.save(cleanLoader, "./imgnet_cleanLoader_imgnet_"+modelPlusList[0].modelName+"_"+modelPlusList[1].modelName)
    # (cleanData, cleanTarget) = torch.load('./Clean_Sun Aug 14 13:47:45 2022.pt')
    # cleanLoader = DMP.TensorToDataLoader(cleanData, cleanTarget, transforms=None, batchSize=10,
    #                                            randomizer=None)

    #Do the attack
    # dataLoaderForTraining = trainLoader
    # epsForAttacks = 0.05
    epsMax = 0.062
    # epsForAttacks = 0.062
    clipMin = 0.0
    clipMax = 1.0
    # clipMin = -1.0
    # clipMax = 1.0
    decayFactor = 1.0
    numSteps = 25
    # numSteps = 80
    # numSteps = 7
    # epsStep = epsForAttacks/numSteps
    # epsStep = 0.005
    epsStep = 0.01
    print('numSteps: ', numSteps, 'epsStep: ',epsStep, 'epsMax: ', epsMax)

    alphaLearningRate = 10000#0#100000
    # alphaLearningRate = 100000#0#100000
    # alphaLearningRate = 10000#0#100000
    fittingFactor = 50.0
    print("Alpha Learning Rate:", alphaLearningRate)
    print("Fitting Factor:", fittingFactor)

    torch.cuda.empty_cache()
    advLoader = AttackWrappersProtoSAGA.SelfAttentionGradientAttackProto(device, epsMax, epsStep, numSteps, modelPlusList,
                                                    cleanLoader, clipMin, clipMax, alphaLearningRate, fittingFactor,advLoader=None, numClasses=numClasses)
    accArrayProtoSAGA = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoader)
        accArrayProtoSAGA = accArrayProtoSAGA + accArray
        print("ProtoSAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_ProtoSAGA_acc = (accArrayProtoSAGA>=1).sum() / numAttackSamples
    print('MV_ProtoSAGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    ALL_MV_ProtoSAGA_acc = (accArrayProtoSAGA==2).sum() / numAttackSamples
    print('ALL_MV_ProtoSAGA_acc: ', ALL_MV_ProtoSAGA_acc.data.cpu().numpy())
    MV_ProtoSAGA_acc = (accArrayProtoSAGA==0).sum() / numAttackSamples
    print('ProtoSAGA attack successful rate: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    # torch.save(advLoaderMIM, saveDir+"//AdvLoaderMIM")
    torch.cuda.empty_cache()


    print()
    print('Basic SAGA attack')
    coefficientArray = torch.zeros(2)
    # secondcoeff = 2.0000e-04
    secondcoeff = 0.5
    coefficientArray[0] = 1.0 - secondcoeff
    coefficientArray[1] = secondcoeff
    print("Coeff Array:")
    print(coefficientArray)
    advLoader = AttackWrappersWhiteBoxP_SAGA.SelfAttentionGradientAttack(device, epsMax, numSteps, epsStep,
                                                modelPlusList, coefficientArray, cleanLoader, clipMin, clipMax, mean, std)
    accArraySAGA = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoader)
        accArraySAGA = accArraySAGA + accArray
        print("SAGA Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_ProtoSAGA_acc = (accArraySAGA>=1).sum() / numAttackSamples
    print('MV_AGA_acc: ', MV_ProtoSAGA_acc.data.cpu().numpy())
    ALL_MV_SAGA_acc = (accArraySAGA>1).sum() / numAttackSamples
    print('ALL_MV_SAGA_acc: ', ALL_MV_SAGA_acc.data.cpu().numpy())
    MV_SAGA_acc = (accArraySAGA==0).sum() / numAttackSamples
    print('SAGA attack successful rate: ', MV_SAGA_acc.data.cpu().numpy())


    print()
    print('MIM attack')
    if ModelPlus_CNN.modelName=='SNN ResNet Backprop':
        advLoaderMIM = AttackWrappersWhiteBoxJelly.MIMNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN,
                                                                  decayFactor, epsMax, epsStep, numSteps, clipMin,
                                                                  clipMax, targeted=False)
    else:
        advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN, decayFactor, epsMax, epsStep,
                                                            numSteps, clipMin, clipMax, targeted=False)
    accArrayMIM_A = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderMIM)
        accArrayMIM_A = accArrayMIM_A + accArray
        print("MIM_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_MIM_A_acc = (accArrayMIM_A>=1).sum() / numAttackSamples
    print('MIM_A_acc: ', MV_MIM_A_acc.data.cpu().numpy())
    ALL_MV_MIM_A_acc = (accArrayMIM_A>1).sum() / numAttackSamples
    print('ALL_MIM_A_acc: ', ALL_MV_MIM_A_acc.data.cpu().numpy())
    ALL_MV_MIM_A_acc = (accArrayMIM_A==0).sum() / numAttackSamples
    print('MIM_A attack successful rate: ', ALL_MV_MIM_A_acc.data.cpu().numpy())
    torch.cuda.empty_cache()


    if ModelPlus_SNN_vgg.modelName=='SNN VGG-16 Backprop':
        advLoaderMIM = AttackWrappersWhiteBoxSNN.MIMNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, decayFactor, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    elif ModelPlus_SNN_vgg.modelName=='SNN ResNet Backprop':
        advLoaderMIM = AttackWrappersWhiteBoxJelly.MIMNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg,
                                                                  decayFactor, epsMax, epsStep, numSteps, clipMin,
                                                                  clipMax, targeted=False)
    else:
        advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, SNN_model_vgg,
                                                                     ModelPlus_SNN_vgg, decayFactor, epsMax, epsStep,
                                                                     numSteps, clipMin, clipMax, targeted=False)
    accArrayMIM_B = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderMIM)
        accArrayMIM_B = accArrayMIM_B + accArray
        print("MIM_B Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_MIM_B_acc = (accArrayMIM_B>=1).sum() / numAttackSamples
    print('MV_MIM_B_acc: ', MV_MIM_B_acc.data.cpu().numpy())
    ALL_MV_MIM_B_acc = (accArrayMIM_B>1).sum() / numAttackSamples
    print('ALL_MV_MIM_B_acc: ', ALL_MV_MIM_B_acc.data.cpu().numpy())
    ALL_MV_MIM_B_acc = (accArrayMIM_B==0).sum() / numAttackSamples
    print('MIM_B attack successful rate: ', ALL_MV_MIM_B_acc.data.cpu().numpy())
    torch.cuda.empty_cache()

    # advLoaderMIM = AttackWrappersWhiteBoxP_SAGA.MIMNativePytorch(device, cleanLoader, SNN_model_res, ModelPlus_SNN_res, decayFactor, epsMax, epsStep,
    #                                                         numSteps, clipMin, clipMax, targeted=False)
    # accArrayMIM_C = torch.zeros(numAttackSamples).to(device)  # Create an array with one entry for ever sample in the dataset
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(advLoaderMIM)
    #     accArrayMIM_C = accArrayMIM_C + accArray
    #     print("MIM_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    # MV_MIM_C_acc = (accArrayMIM_C>=1).sum() / numAttackSamples
    # print('MV_MIM_C_acc: ', MV_MIM_C_acc.data.cpu().numpy())
    # ALL_MV_MIM_C_acc = (accArrayMIM_C>1).sum() / numAttackSamples
    # print('ALL_MV_MIM_C_acc: ', ALL_MV_MIM_C_acc.data.cpu().numpy())
    # ALL_MV_MIM_C_acc = (accArrayMIM_C==0).sum() / numAttackSamples
    # print('MIM_C attack successful rate: ', ALL_MV_MIM_C_acc.data.cpu().numpy())
    torch.cuda.empty_cache()
    print()
    print('MV_MIM_acc: ', torch.min(MV_MIM_A_acc,MV_MIM_B_acc).data.cpu().numpy())
    # print('MV_MIM_acc: ', torch.min(torch.min(MV_MIM_A_acc,MV_MIM_B_acc), MV_MIM_C_acc).data.cpu().numpy())
    # print('ALL_MV_MIM_acc: ', torch.min(torch.min(ALL_MV_MIM_A_acc,ALL_MV_MIM_B_acc), ALL_MV_MIM_C_acc).data.cpu().numpy())
    print()

    print('PGD attack')
    if ModelPlus_CNN.modelName == 'SNN ResNet Backprop':
        advLoaderPGD = AttackWrappersWhiteBoxJelly.PGDNativePytorch(device, cleanLoader, syntheticModel,
                                                                    ModelPlus_CNN, epsMax, epsStep, numSteps,
                                                                    clipMin, clipMax, targeted=False)
    else:
        advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, syntheticModel, ModelPlus_CNN, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    accArrayPGD_A = torch.zeros(numAttackSamples).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_A = accArrayPGD_A + accArray
        print("PGD_A Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_PGD_A_acc = (accArrayPGD_A>=1).sum() / numAttackSamples
    print('MV_PGD_A_acc: ', MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A>1).sum() / numAttackSamples
    print('ALL_MV_PGD_A_acc: ', ALL_MV_PGD_A_acc.data.cpu().numpy())
    ALL_MV_PGD_A_acc = (accArrayPGD_A==0).sum() / numAttackSamples
    print('PGD_A attack successful rate: ', ALL_MV_PGD_A_acc.data.cpu().numpy())

    if ModelPlus_SNN_vgg.modelName=='SNN VGG-16 Backprop':
        advLoaderPGD = AttackWrappersWhiteBoxSNN.PGDNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    elif ModelPlus_SNN_vgg.modelName=='SNN ResNet Backprop':
        advLoaderPGD = AttackWrappersWhiteBoxJelly.PGDNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    else:
        advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, SNN_model_vgg, ModelPlus_SNN_vgg, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    accArrayPGD_B = torch.zeros(numAttackSamples).to(device)
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(advLoaderPGD)
        accArrayPGD_B = accArrayPGD_B + accArray
        print("PGD_B Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    MV_PGD_B_acc = (accArrayPGD_B>=1).sum() / numAttackSamples
    print('MV_PGD_B_acc: ', MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B>1).sum() / numAttackSamples
    print('ALL_MV_PGD_B_acc: ', ALL_MV_PGD_B_acc.data.cpu().numpy())
    ALL_MV_PGD_B_acc = (accArrayPGD_B==0).sum() / numAttackSamples
    print('PGD_B attack successful rate: ', ALL_MV_PGD_B_acc.data.cpu().numpy())

    # advLoaderPGD = AttackWrappersWhiteBoxP_SAGA.PGDNativePytorch(device, cleanLoader, SNN_model_res, ModelPlus_SNN_res, epsMax, epsStep, numSteps, clipMin, clipMax, targeted=False)
    # accArrayPGD_C = torch.zeros(numAttackSamples).to(device)
    # for i in range(0, numModels):
    #     accArray = modelPlusList[i].validateDA(advLoaderPGD)
    #     accArrayPGD_C = accArrayPGD_C + accArray
    #     print("PGD_C Acc " + modelPlusList[i].modelName + ":", accArray.sum()/numAttackSamples)
    # MV_PGD_C_Acc = (accArrayPGD_C>=2).sum() / numAttackSamples
    # print('MV_PGD_C_Acc: ', MV_PGD_C_Acc.data.cpu().numpy())
    # ALL_MV_PGD_C_Acc = (accArrayPGD_C>2).sum() / numAttackSamples
    # print('ALL_MV_PGD_C_Acc: ', ALL_MV_PGD_C_Acc.data.cpu().numpy())
    # ALL_MV_PGD_C_Acc = (accArrayPGD_C==0).sum() / numAttackSamples
    # print('PGD_C attack successful rate: ', ALL_MV_PGD_C_Acc.data.cpu().numpy())

    print()
    print('MV_PGD_acc: ', torch.min(MV_PGD_A_acc,MV_PGD_B_acc).data.cpu().numpy())
    # print('MV_PGD_acc: ', torch.min(torch.min(MV_PGD_A_acc,MV_PGD_B_acc), MV_PGD_C_Acc).data.cpu().numpy())
    # print('ALL_MV_PGD_acc: ', torch.min(torch.min(ALL_MV_PGD_A_acc,ALL_MV_PGD_B_acc), ALL_MV_PGD_C_Acc).data.cpu().numpy())
    print()
    # #Go through and check the robust accuray of each model on the adversarial examples
    # for i in range(0, len(modelPlusList)):
    #     acc = modelPlusList[i].validateD(advLoader)
    #     print(modelPlusList[i].modelName+" Robust Acc:", acc)




#Load the BiT-M-R50x1
def LoadBiT_R50():
    #Basic variable and data setup
    modelPlusList = []
    device = torch.device("cuda")
    numClasses = 10
    # imgSize = 224
    batchSize = 8
    threshold = 1
    #Load the CIFAR-10 data
    valLoader = DMP.GetCIFAR10Validation_unnormalize_resize(imgSizeH=160, imgSizeW=128, batchSize=64, mean=0.5)
    #Load the BiT-M-R101x3
    dir = "./BiT-M-R50x1-Run0.tar"
    model = BigTransferModels.KNOWN_MODELS["BiT-M-R50x1"](head_size=numClasses, zero_head=False)
    #Get the checkpoint 
    checkpoint = torch.load(dir, map_location="cpu")
    #Remove module so that it will load properly
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #Load the dictionary
    model.load_state_dict(new_state_dict)
    print("load model from ", dir)
    model.eval()
    #Wrap the model in the ModelPlus class
    #Here we hard code the Big Transfer Model Plus class input size to 160x128 (what it was trained on)
    modelBig50Plus = ModelPlus.ModelPlus("BiT-M-R50x1", model, device, imgSizeH=160, imgSizeW=128, batchSize=batchSize)
    modelPlusList.append(modelBig50Plus)
    # acc = modelBig50Plus.validateD(valLoader)
    # print("Model accuracy: ", acc)
    defense = None
    return model, modelBig50Plus, defense, valLoader


#Load the spiking jelly resnet snn
def LoadsjSNNresnet(arch, device, batchSize):
    #Basic variable and data setup
    modelPlusList = []
    # device = torch.device("cuda")
    # device = torch.device("cpu")
    numClasses = 1000
    # numClasses = 10
    imgSize = 224
    batchSize = batchSize
    threshold = 1
    # valLoader = DMP.GetCIFAR100Validation_resize(imgSizeH=160, imgSizeW=128, batchSize=batchSize)
    #Load the BiT-M-R101x3
    # dir = "./BiT-M-R101x3-Run0.tar"

    if arch == 'sew_resnet18':
        dir = "/home/nux219/temporal_snn/attack/logs/b64_e90_sgd_lr0.1_wd0_m0.9_cosa90_pt/ckp_latest.pt"
        print('load SNN model from ', dir)
        model = sew_resnet.multi_step_sew_resnet18(T=4, pretrained=False, multi_step_neuron=neuron.MultiStepIFNode, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy', cnf='ADD')
        savedVar = torch.load(dir, map_location='cpu')
        model.load_state_dict(savedVar['model'], strict= False)
    elif arch == 'spiking_resnet18':
        dir = "/home/nux219/temporal_snn/attack/logs_vanilla_spiking_resnet/b64_e90_sgd_lr0.1_wd0_m0.9_cosa90_pt/ckp_latest.pt"
        print('load SNN model from ', dir)
        model = spiking_resnet.multi_step_spiking_resnet18(T=4, pretrained=False,
                                                         multi_step_neuron=neuron.MultiStepIFNode,
                                                         surrogate_function=surrogate.ATan(), detach_reset=True,
                                                         backend='cupy')
        savedVar = torch.load(dir, map_location='cpu')
        model.load_state_dict(savedVar['model'], strict= False)
    # model = nn.DataParallel(model)
    model.eval()
    # model = model.to(device)
    #Wrap the model in the ModelPlus class
    #Here we hard code the Big Transfer Model Plus class input size to 512 (what it was trained on)
    modelPlus = ModelPlus.ModelPlusSpikingJelly("SNN ResNet Backprop", model, device, imgSize, imgSize, batchSize)
    # modelPlusList.append(modelPlus)
    # acc = modelBig101Plus.validateD(valLoader)
    # print("Model accuracy: ", acc)
    # defense = None
    return model, modelPlus


#Load the BiT-M-R152x4
def LoadBiT_R152(device, batchSize512):
    #Basic variable and data setup
    modelPlusList = []
    # device = torch.device("cuda")
    # device = torch.device("cpu")
    numClasses = 1000
    # numClasses = 10
    imgSize = 512
    batchSize = batchSize512
    threshold = 1
    # valLoader = DMP.GetCIFAR100Validation_resize(imgSizeH=160, imgSizeW=128, batchSize=batchSize)
    #Load the BiT-M-R101x3
    # dir = "./BiT-M-R101x3-Run0.tar"
    dir = "/home/nux219/BarrierZoneTrainer/imagenet_models/BiT-M-R152x4-ILSVRC2012.npz"
    print('load BiT model from ', dir)
    modelBig152 = BigTransferModels.KNOWN_MODELS["BiT-M-R152x4"](head_size=1000, zero_head=False, device=device)
    #Get the checkpoint 
    modelBig152.load_from(np.load(dir))
    # modelBig152 = nn.DataParallel(modelBig152)
    modelBig152.eval()
    # modelBig152 = modelBig152.to(device)
    #Wrap the model in the ModelPlus class
    #Here we hard code the Big Transfer Model Plus class input size to 512 (what it was trained on)
    modelBig152Plus = ModelPlus.ModelPlus("BiT-M-R152x4", modelBig152, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
    modelPlusList.append(modelBig152Plus)
    # acc = modelBig101Plus.validateD(valLoader)
    # print("Model accuracy: ", acc)
    defense = None
    return modelBig152, modelBig152Plus, defense


#Load the VGGSNN_bp
def LoadVGG16SNN_bp(batchSize):
    #Load the Backprop SNN VGG-16
    # dirC = "./pretrained_cifar10_vgg_3_acc_8919_ver1"
    dirC = "./pretrained_cifar100_vgg_1_acc_6396_ver1"
    print('load SNN vgg16 BP from ', dirC)
    # length = 20
    length = 30
    tau_m = 8
    tau_s = 2
    imgSize = 32
    device = torch.device("cuda")
    train_synapse_tau = True
    train_neuron_tau = True
    membrane_filter = True
    neuron_model = 'iir'
    synapse_type = 'first_order_low_pass' # first_order_low_pass or dual_exp or none
    dropout = 0.2
    modelBackpropSNN = SNNBackpropModel.SNNBackprop_grad_estimation(length, batchSize, tau_m, tau_s, train_synapse_tau, train_neuron_tau, membrane_filter, neuron_model, synapse_type, dropout)
    savedVar = torch.load(dirC)
    modelBackpropSNN.load_state_dict(savedVar['snn_state_dict'], strict= False)
    modelBackpropSNN.eval()
    modelBackpropSNN.change_threshold_func(snn_functions.threshold_arctan) #Changes to the best setting for generating adversarial examples, does not effect forward pass clean acc
    modelPlusSNNBackprop = ModelPlus.ModelPlusSNNRepeat("SNN VGG-16 Backprop", modelBackpropSNN, device, 32, 32, batchSize, length)
    # valLoader_snn = DMP.GetCIFAR10Validation_snn(imgSize, batchSize, length=length)
    valLoader_snn = DMP.GetCIFAR100Validation_snn(imgSize, batchSize, length=length)
    return modelBackpropSNN, modelPlusSNNBackprop, valLoader_snn


#Load the ViT-L-16
def LoadViTL():
    #Basic variable and data setup
    modelPlusList = []
    device = torch.device("cuda")
    # numClasses = 10
    numClasses = 100
    imgSize = 224
    batchSize = 10
    threshold = 1
    #Load the CIFAR-10 data
    # valLoader = DMP.GetCIFAR10Validation_unnormalize(imgSize, batchSize=32)
    valLoader = DMP.GetCIFAR100Validation(imgSize, batchSize=32)
    #Load ViT-L-16
    config = CONFIGS["ViT-L_16"]
    model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses, vis=True)
    # dir = "./ViT-L_16,cifar10,run0_15K_checkpoint.bin"
    dir = "./ViT-L_16,cifar100,run0_15K_checkpoint.bin"
    dict = torch.load(dir)
    model.load_state_dict(dict)
    print("load model from ", dir)
    model.eval()
    #Wrap the model in the ModelPlus class
    modelPlus_model = ModelPlus.ModelPlus("ViT-L_16", model, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
    modelPlusList.append(modelPlus_model)
    # acc = modelPlus_model.validateD(valLoader)
    # print("Model accuracy: ", acc)
    defense = None
    return model, modelPlus_model, defense, valLoader


#Load the ViT-L-16 imagenet
def LoadViTL_imagenet(device, imgSize):
    #Basic variable and data setup
    modelPlusList = []
    # device = torch.device("cuda")
    # device = torch.device("cpu")
    numClasses = 1000
    # imgSize = 224
    # imgSize = 512
    batchSize = 32
    threshold = 1
    #Load the CIFAR-10 data
    # valLoader = DMP.GetCIFAR10Validation_unnormalize(imgSize, batchSize=32)
    # valLoader = DMP.GetIMAGENETValidation(imgSize, batchSize=batchSize)
    valLoader = None
    #Load ViT-L-16
    config = CONFIGS["ViT-L_16"]
    model = VisionTransformer(config, imgSize, zero_head=False, num_classes=numClasses, vis=True, device=device)
    # dir = "./ViT-L_16,cifar10,run0_15K_checkpoint.bin"
    if imgSize == 224:
        dir = "/home/nux219/BarrierZoneTrainer/imagenet_models/imagenet21k+imagenet2012_ViT-L_16-224.npz"
        # dir = "/home/nux219/BarrierZoneTrainer/imagenet_models/imagenet21k+imagenet2012_ViT-L_16.npz"
        model.load_from(np.load(dir))
        print("load model from ", dir)
        model.eval()
        # model = model.to(device)
        #Wrap the model in the ModelPlus class
        modelPlus_model = ModelPlus.ModelPlus("ViT-L_16-224", model, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
    else:
        dir = "/home/nux219/BarrierZoneTrainer/imagenet_models/imagenet21k+imagenet2012_ViT-L_16.npz"
        model.load_from(np.load(dir))
        print("load model from ", dir)
        model.eval()
        # model = model.to(device)
        #Wrap the model in the ModelPlus class
        modelPlus_model = ModelPlus.ModelPlus("ViT-L_16-512", model, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
    modelPlusList.append(modelPlus_model)
    # acc = modelPlus_model.validateD(valLoader)
    # print("Model accuracy: ", acc)
    defense = None
    return model, modelPlus_model, defense, valLoader


#Load the ViT-B-16
def LoadViTB():
    #Basic variable and data setup
    modelPlusList = []
    device = torch.device("cuda")
    numClasses = 10
    imgSize = 224
    batchSize = 32
    threshold = 1
    #Load the CIFAR-10 data
    valLoader = DMP.GetCIFAR10Validation_unnormalize(imgSize, batchSize)
    #Load ViT-L-16
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses, vis=True)
    dir = "./ViT-B_16,cifar10,run0_checkpoint.bin"
    dict = torch.load(dir)
    model.load_state_dict(dict)
    print("load model from ", dir)
    model.eval()
    #Wrap the model in the ModelPlus class
    modelPlus_model = ModelPlus.ModelPlus("ViT-B_16", model, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
    modelPlusList.append(modelPlus_model)
    # acc = modelPlus.validateD(valLoader)
    # print("Model accuracy: ", acc)
    defense = None
    return model, modelPlus_model, defense, valLoader




def Loadresnet(modelDir):
    #List to hold the models
    modelPlusList = []
    #Model parameters
    numClasses = 10
    inputImageSize = 32
    batchSize = 32
    threshold = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load resnet model
    model = resnet56.resnet56(32, numClasses)
    # model = resnet56.resnet164(32, numClasses)
    checkpoint = torch.load(modelDir)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    #Append the current model to the model list
    modelPlusList.append(ModelPlus.ModelPlus("ModelResNet56", model, device, inputImageSize, inputImageSize, batchSize))
    ModelPlus_model = ModelPlus.ModelPlus("ModelResNet56", model, device, inputImageSize, inputImageSize, batchSize)
    #Call the constructor for the BARZ defense
    defense = None
    return defense, ModelPlus_model, model





def LoadDietSNN_resnet(modelDir, datasets, batchsize, mean, std):
    modelPlusList = []
    architecture        = 'RESNET20'
    # timesteps           = 20

    leak                = 1.0
    # default_threshold   = 1.0
    default_threshold   = 0.4
    activation          = 'Linear'
    kernel_size         = 3
    dataset             = datasets
    if dataset == 'CIFAR10':
        labels              = 10
        numClasses = 10
        timesteps = 10
    else:
        labels              = 100
        numClasses = 100
        timesteps = 8
    inputImageSize = 32
    batchSize = batchsize
    threshold = 1
    alpha=0.3
    beta = 0.01
    dropout = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if architecture[0:3].lower() == 'res':
        # model = resnet_spiking_diet.RESNET_SNN_STDB(resnet_name=architecture, activation=activation, labels=labels, timesteps=timesteps,
        #                         leak=leak, default_threshold=default_threshold, alpha=alpha, beta=beta, dropout=dropout,
        #                         dataset=dataset)
        model = resnet_spiking_SAGA.RESNET_SNN_STDB(resnet_name=architecture, activation=activation, labels=labels, timesteps=timesteps,
                                                    leak=leak, default_threshold=default_threshold, alpha=alpha, beta=beta, dropout=dropout,
                                                    dataset=dataset, mean=mean, std=std)
    # model = nn.DataParallel(model)
    state = torch.load(modelDir, map_location='cpu')
    print('load SNN from modelDir: ', modelDir)
    missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
    print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
    # model.module.network_update(timesteps=timesteps, leak=leak)
    model.network_update(timesteps=timesteps, leak=leak)
    model.eval()
    temp1 = []
    temp2 = []
    # for key, value in sorted(model.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
    for key, value in sorted(model.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
        temp1 = temp1 + [round(value.item(), 2)]
    # for key, value in sorted(model.module.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
    for key, value in sorted(model.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
        temp2 = temp2 + [round(value.item(), 2)]
    print('\n Thresholds: {}, leak: {}'.format(temp1, temp2))
    #Append the current model to the model list
    modelPlusList.append(ModelPlus.ModelPlus("Trans_SNN_resnet", model, device, inputImageSize, inputImageSize, batchSize))
    #Call the constructor for the BARZ defense
    ModelPlus_SNN = ModelPlus.ModelPlus("Trans_SNN_resnet", model, device, inputImageSize, inputImageSize, batchSize)
    return ModelPlus_SNN, model


def LoadDietSNN_imgnet(modelDir, batchsize, imgSize, device, timesteps=5):
    modelPlusList = []
    architecture        = 'VGG16'
    # architecture        = 'VGG11'
    # timesteps           = 5
    # timesteps           = 8
    timesteps           = timesteps
    leak                = 1.0
    default_threshold   = 1.0
    # default_threshold   = 1.0
    activation          = 'Linear'
    kernel_size         = 3
    labels              = 1000
    numClasses = 1000
    inputImageSize = 224
    batchSize = batchsize
    threshold = 1
    scaling_factor = 1
    dataset = 'IMAGENET'
    print('number of class: ', labels)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = vgg_spiking_diet.VGG_SNN_STDB(vgg_name=architecture, activation=activation, labels=labels, timesteps=timesteps, leak=leak,
    #                 default_threshold=default_threshold, dropout=0.2, kernel_size=kernel_size, dataset=dataset)
    model = vgg_spiking_imagenet.VGG_SNN_STDB_IMAGENET(vgg_name=architecture, activation=activation, labels=labels, timesteps=timesteps,
                                                       leak=leak, default_threshold=default_threshold, alpha=0.3, beta=0.01,
                                                       dropout=0.2, kernel_size=kernel_size, dataset=dataset, device=device)

    model = nn.DataParallel(model)
    state = torch.load(modelDir, map_location='cpu')
    print('load SNN from modelDir: ', modelDir)
    # cur_dict = model.state_dict()
    missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
    print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
    # model.to(device)
    model.eval()
    #Append the current model to the model list
    modelPlusList.append(ModelPlus.ModelPlus("Trans_SNN_vgg_"+str(timesteps), model, device, inputImageSize, inputImageSize, batchSize))
    ModelPlus_SNN = ModelPlus.ModelPlus("Trans_SNN_vgg_"+str(timesteps), model, device, inputImageSize, inputImageSize, batchSize)
    return ModelPlus_SNN, model



def LoadDietSNN(modelDir, datasets, batchsize, mean, std, timesteps=10):
    modelPlusList = []
    architecture        = 'VGG16'
    # architecture        = 'VGG11'
    # timesteps           = 5
    # timesteps           = 8
    timesteps           = timesteps
    leak                = 1.0
    default_threshold   = 0.4
    # default_threshold   = 1.0
    activation          = 'Linear'
    kernel_size         = 3
    dataset             = datasets
    if dataset == 'CIFAR10':
        labels              = 10
        numClasses = 10
    else:
        labels              = 100
        numClasses = 100
    inputImageSize = 32
    batchSize = batchsize
    threshold = 1
    scaling_factor = 0.2
    print('number of class: ', labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = vgg_spiking_diet.VGG_SNN_STDB(vgg_name=architecture, activation=activation, labels=labels, timesteps=timesteps, leak=leak,
    #                 default_threshold=default_threshold, dropout=0.2, kernel_size=kernel_size, dataset=dataset)
    model = vgg_spiking_SAGA.VGG_SNN_STDB(vgg_name=architecture, activation=activation, labels=labels,
                                          timesteps=timesteps, leak=leak,
                                          default_threshold=default_threshold, dropout=0.2, kernel_size=kernel_size,
                                          dataset=dataset, mean=mean, std=std)
    # model = nn.DataParallel(model)
    state = torch.load(modelDir, map_location='cpu')
    print('load SNN from modelDir: ', modelDir)
    # cur_dict = model.state_dict()
    # for key in state['state_dict'].keys():
    #     # if key in cur_dict:
    #     if key.strip('module.') in cur_dict:
    #         # if (state['state_dict'][key].shape == cur_dict[key].shape):
    #         #     cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
    #         #     print('\n Loaded {} from {}'.format(key, modelDir))
    #         if (state['state_dict'][key].shape == cur_dict[key.strip('module.')].shape):
    #             cur_dict[key.strip('module.')] = nn.Parameter(state['state_dict'][key].data)
    #             print('\n Loaded {} from {}'.format(key.strip('module.'), modelDir))
    #         else:
    #             print('\n Size mismatch {}, size of loaded model {}, size of current model {}'.format(key,
    #                                                                                                     state['state_dict'][
    #                                                                                                         key].shape,
    #                                                                                                     model.state_dict()[
    #                                                                                                       key].shape))
    #     elif key.strip('encoder.') in cur_dict:
    #         cur_dict['encoder.threshold'] = nn.Parameter(state['state_dict'][key].data)
    #         print('\n Loaded {} from {}'.format('encoder.threshold', modelDir))
    #
    #     else:
    #         print('\n Loaded weight {} not present in current model'.format(key))
    # model.load_state_dict(cur_dict)
    # missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
    print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
    model.eval()
    #Append the current model to the model list
    modelPlusList.append(ModelPlus.ModelPlus("Trans_SNN_vgg_"+str(timesteps), model, device, inputImageSize, inputImageSize, batchSize))
    #Call the constructor for the BARZ defense
    ModelPlus_SNN = ModelPlus.ModelPlus("Trans_SNN_vgg_"+str(timesteps), model, device, inputImageSize, inputImageSize, batchSize)
    return ModelPlus_SNN, model







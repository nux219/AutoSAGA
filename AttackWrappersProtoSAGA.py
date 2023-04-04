import os
import torch
import DataManagerPytorch as DMP
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2
import time
from spikingjelly.clock_driven import functional

# Try to replicate the momentum of AutoAttack without implmenting the decreased learning rate
def SemiAutoAttackV2(device, dataLoader, model, epsilonMax, numSteps, clipMin, clipMax):
    print("Warning this is hard coded semi-auto attack. Do not call outside of debugging purposes.")
    model.eval()  # Change model to evaluation mode for the attack
    # Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  # just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    loss2 = torch.nn.CrossEntropyLoss(reduction='none')
    tracker = 0
    epsilonStep = epsilonMax / float(numSteps)
    # Autoattack stuff
    checkPointList = ComputeAutoAttackCheckpoints(numSteps)
    a = 0.75
    # Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        # Put the data from the batch onto the device
        x = xData.to(device)
        xk = xData.to(device)
        y = yData.type(torch.LongTensor).to(device)
        ###Autoattack
        # lossFunctionCE = torch.nn.CrossEntropyLoss(reduction='none')
        logits = model(xk)
        fMax = loss2(logits, y)
        xMax = xData.clone()
        # Do the attack for a number of steps
        for attackStep in range(0, numSteps):
            # The first step is different from the rest
            if attackStep == 0:
                xk.requires_grad = True
                outputs = model(xk)
                model.zero_grad()
                cost = loss(outputs, y).to(device)
                cost.backward()
                xk = xk + (epsilonStep * xk.grad.data.sign())
                # Adding clipping to maintain the range
                xk = torch.clamp(x, min=clipMin, max=clipMax)
                xkMinus1 = x.clone()
            else:
                # Part A: Compute zk
                xk.requires_grad = True
                outputs = model(xk)
                model.zero_grad()
                cost = loss(outputs, y).to(device)
                cost.backward()
                zk1 = xk + (epsilonStep * xk.grad.data.sign()).to(device)  # Internal computation of zk
                zk1 = ProjectionS(zk1, x, epsilonMax, clipMin, clipMax)
                # Part B: Compute x(k+1)
                xk1 = xk + a * (zk1 - xk) + (1 - a) * (xk - xkMinus1)
                xk1 = ProjectionS(xk1, x, epsilonMax, clipMin, clipMax)
                # Part C: Set variables for the next round
                xkMinus1 = xk.detach_()  # kMinus1 <= xk
                xk = xk1.detach_()  # xk <=x(k+1)
                # This is part where we do the checks
                print(xk.shape)
                logits = model(xk1)
                fCurrent = loss2(logits, y).to(torch.device("cpu"))
                for i in range(0, batchSize):
                    if fCurrent[i] >= fMax[i]:
                        fMax[i] = fCurrent[i]
                        xMax[i] = xk[i]
        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xMax[j]  # xk1[j]
            yClean[advSampleIndex] = y[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
    # All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,
                                       randomizer=None)  # use the same batch size as the original loader
    return advLoader


# This is terribly coded
def ComputeAutoAttackCheckpoints(nIter):
    # P list according to the paper
    p = []
    p.append(0)
    p.append(0.22)
    for j in range(1, nIter):
        pCurrent = p[j] + max(p[j] - p[j - 1] - 0.03, 0.06)
        if np.ceil(pCurrent * nIter) <= nIter:
            p.append(pCurrent)
        else:
            break
    # After we make p list can compute the actual checkpoints w[j]
    w = []
    for j in range(0, len(p)):
        w.append(int(np.ceil(p[j] * nIter)))
    # return checkpoints w
    return w


# Try to replicate the momentum of AutoAttack without implmenting the decreased learning rate
def SemiAutoAttack(device, dataLoader, model, epsilonMax, numSteps, clipMin, clipMax):
    print("Warning this is hard coded semi-auto attack. Do not call outside of debugging purposes.")
    model.eval()  # Change model to evaluation mode for the attack
    # Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  # just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    epsilonStep = epsilonMax / float(numSteps)
    a = 0.75
    # Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        # Put the data from the batch onto the device
        x = xData.to(device)
        xk = xData.to(device)
        y = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        # Initalize memory for the gradient momentum
        # Do the attack for a number of steps
        for attackStep in range(0, numSteps):
            # The first step is different from the rest
            if attackStep == 0:
                xk.requires_grad = True
                outputs = model(xk)
                model.zero_grad()
                cost = loss(outputs, y).to(device)
                cost.backward()
                xk = xk + (epsilonStep * xk.grad.data.sign())
                # Adding clipping to maintain the range
                xk = torch.clamp(x, min=clipMin, max=clipMax)
                xkMinus1 = x.clone()
            else:
                # Part A: Compute zk
                xk.requires_grad = True
                outputs = model(xk)
                model.zero_grad()
                cost = loss(outputs, y).to(device)
                cost.backward()
                zk1 = xk + (epsilonStep * xk.grad.data.sign()).to(device)  # Internal computation of zk
                zk1 = ProjectionS(zk1, x, epsilonMax, clipMin, clipMax)
                # Part B: Compute x(k+1)
                xk1 = xk + a * (zk1 - xk) + (1 - a) * (xk - xkMinus1)
                xk1 = ProjectionS(xk1, x, epsilonMax, clipMin, clipMax)
                # Part C: Set variables for the next round
                xkMinus1 = xk.detach_()  # kMinus1 <= xk
                xk = xk1.detach_()  # xk <=x(k+1)
        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xk1[j]
            yClean[advSampleIndex] = y[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
    # All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,
                                       randomizer=None)  # use the same batch size as the original loader
    return advLoader


# Projection operation
def ProjectionS(xAdv, x, epsilonMax, clipMin, clipMax):
    return torch.clamp(torch.min(torch.max(xAdv, x - epsilonMax), x + epsilonMax), clipMin, clipMax)


# Try to replicate part of the AutoAttack
def PGDNativeAttack(device, dataLoader, model, epsilonMax, numSteps, clipMin, clipMax, targeted):
    model.eval()  # Change model to evaluation mode for the attack
    # Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  # just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    epsilonStep = epsilonMax / float(numSteps)
    # Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        # Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        # Initalize memory for the gradient momentum
        # Do the attack for a number of steps
        for attackStep in range(0, numSteps):
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            advTemp = xAdvCurrent + (epsilonStep * xAdvCurrent.grad.data.sign()).to(device)
            # Adding clipping to maintain the range
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
    # All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,
                                       randomizer=None)  # use the same batch size as the original loader
    return advLoader


# Main attack method, takes in a list of models and a clean data loader
# Returns a dataloader with the adverarial samples and corresponding clean labels
def SelfAttentionGradientAttackProtoAuto(device, epsMax, numSteps, modelListPlus, dataLoader, clipMin, clipMax,
                                         alphaLearningRate, fittingFactor):
    print("Using hard coded experimental function not advisable.")
    # Basic graident variable setup
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    xAdv = xClean  # Set the initial adversarial samples
    xOridata = xClean.to(device).detach()
    xOriMax = xOridata + epsMax
    xOriMin = xOridata - epsMax
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    # Compute eps step
    epsStep = epsMax / numSteps
    dataLoaderCurrent = dataLoader
    # Hardcoded for alpha right now, put in the method later
    confidence = 0
    nClasses = 10
    alpha = torch.ones(len(modelListPlus), numSamples, xShape[0], xShape[1],
                       xShape[2])  # alpha for every model and every sample
    # End alpha setup
    numSteps = 10
    for i in range(0, numSteps):
        print("Running step", i)
        # Keep track of dC/dX for each model where C is the cross entropy function
        dCdX = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        # Keep track of dF/dX for each model where F, is the Carlini-Wagner loss function (for updating alpha)
        dFdX = torch.zeros(numSamples, xShape[0], xShape[1],
                           xShape[2])  # Change to the math here to take in account all objecitve functions
        # Go through each model and compute dC/dX
        for m in range(0, len(modelListPlus)):
            dataLoaderCurrent = modelListPlus[m].formatDataLoader(dataLoaderCurrent)
            dCdXTemp = FGSMNativeGradient(device, dataLoaderCurrent, modelListPlus[m])
            # Resize the graident to be the correct size and save it
            dCdX[m] = torch.nn.functional.interpolate(dCdXTemp, size=(xShape[1], xShape[2]))
            # Now compute the inital adversarial example with the base alpha
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            xGradientCumulative = xGradientCumulative + alpha[m] * dCdX[m]
        xAdvStepOne = xAdv + epsStep * xGradientCumulative.sign()
        # Convert the current xAdv to dataloader
        dataLoaderStepOne = DMP.TensorToDataLoader(xAdvStepOne, yClean, transforms=None,
                                                   batchSize=dataLoader.batch_size, randomizer=None)
        print("===Pre-Alpha Optimization===")
        costMultiplier = torch.zeros(len(modelListPlus), numSamples)
        for m in range(0, len(modelListPlus)):
            cost = CheckCarliniLoss(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses)
            costMultiplier[m] = CarliniSingleSampleLoss(device, dataLoaderStepOne, modelListPlus[m], confidence,
                                                        nClasses)
            print("For model", m, "the Carlini value is", cost)
        # Compute dF/dX (cumulative)
        for m in range(0, len(modelListPlus)):
            dFdX = dFdX + torch.nn.functional.interpolate(
                dFdXCompute(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses),
                size=(xShape[1], xShape[2]))
        # Compute dX/dAlpha
        dXdAlpha = dXdAlphaCompute(fittingFactor, epsStep, alpha, dCdX, len(modelListPlus), numSamples, xShape)
        # Compute dF/dAlpha = dF/dx * dX/dAlpha
        dFdAlpha = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            # dFdAlpha = dFdX * dXdAlpha[m]
            dFdAlpha[m] = dFdX * dXdAlpha[m]
        # Now time to update alpha
        alpha = alpha - dFdAlpha * alphaLearningRate
        # Compute final adversarial example using best alpha
        xGradientCumulativeB = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            xGradientCumulativeB = xGradientCumulativeB + alpha[m] * dCdX[m]
        xAdv = xAdv + epsStep * xGradientCumulativeB.sign()
        # clipMin = 0.0
        # clipMax = 1.0
        xAdv = torch.min(xOridata + epsMax, xAdv)
        xAdv = torch.max(xOridata - epsMax, xAdv)
        xAdv = torch.clamp(xAdv, clipMin, clipMax)
        # Convert the current xAdv to dataloader
        dataLoaderCurrent = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,
                                                   randomizer=None)
        # Debug HERE
        print("===Post-Alpha Optimization===")
        for m in range(0, len(modelListPlus)):
            cost = CheckCarliniLoss(device, dataLoaderCurrent, modelListPlus[m], confidence, nClasses)
            print("For model", m, "the Carlini value is", cost)
    return dataLoaderCurrent


# Main attack method, takes in a list of models and a clean data loader
# Returns a dataloader with the adverarial samples and corresponding clean labels
def SelfAttentionGradientAttack(device, epsMax, numSteps, epsStep, modelListPlus, coefficientArray, dataLoader, clipMin,
                                clipMax, mean, std):
    # Basic graident variable setup
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    xAdv = xClean.to(device).detach()  # Set the initial adversarial samples
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    xOridata = xClean.to(device).detach()

    # Compute eps step
    # epsStep = epsMax / numSteps
    dataLoaderCurrent = dataLoader
    for i in range(0, numSteps):
        print("Running Step=", i)
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            currentModelPlus = modelListPlus[m]
            # First get the gradient from the model
            # dataLoaderCurrent = currentModelPlus.formatDataLoader(dataLoaderCurrent)
            xGradientCurrent = FGSMNativeGradient(device, dataLoaderCurrent, currentModelPlus)
            # Resize the graident to be the correct size
            xGradientCurrent = torch.nn.functional.interpolate(xGradientCurrent, size=(xShape[1], xShape[2]))
            # Add the current computed gradient to the result
            if currentModelPlus.modelName.find("ViT") >= 0:
                attmap = GetAttention(dataLoaderCurrent, currentModelPlus)
                attmap = torch.nn.functional.interpolate(attmap, size=(xShape[1], xShape[2]))
                xGradientCumulative = xGradientCumulative + coefficientArray[m] * xGradientCurrent * attmap
            else:
                xGradientCumulative = xGradientCumulative + coefficientArray[m] * xGradientCurrent
        # Compute the sign of the graident and create the adversarial example
        # xAdv = xAdv + epsStep * xGradientCumulative.sign()
        xAdv = xAdv + epsStep * xGradientCumulative.to(device).sign()
        xAdv = torch.min(xOridata + epsMax, xAdv)
        xAdv = torch.max(xOridata - epsMax, xAdv)
        # Do the clipping
        xAdv = torch.clamp(xAdv, clipMin, clipMax)
        # Convert the result to dataloader
        dataLoaderCurrent = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,
                                                   randomizer=None)
    return dataLoaderCurrent


# Main attack method, takes in a list of models and a clean data loader
# Returns a dataloader with the adverarial samples and corresponding clean labels
def SelfAttentionGradientAttackProto(device, epsMax, epsStep, numSteps, modelListPlus, dataLoader, clipMin, clipMax,
                                     alphaLearningRate, fittingFactor, advLoader=None, numClasses=10):
    print("Using hard coded experimental function not advisable.")
    # Basic graident variable setup
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    if advLoader is not None:
        xAdv, _ = DMP.DataLoaderToTensor(advLoader)
        dataLoaderCurrent = advLoader
    else:
        xAdv = xClean  # Set the initial adversarial samples
        dataLoaderCurrent = dataLoader
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    xOridata = xClean.detach()
    xOriMax = xOridata + epsMax
    xOriMin = xOridata - epsMax
    print('input size: ', xClean.shape, xShape)
    # Compute eps step
    # epsStep = epsMax / numSteps

    # Hardcoded for alpha right now, put in the method later
    confidence = 0
    nClasses = numClasses
    alpha = torch.ones(len(modelListPlus), numSamples, xShape[0], xShape[1],
                       xShape[2])  # alpha for every model and every sample
    # End alpha setup
    # numSteps = 10
    for i in range(0, numSteps):
        print("Running step", i)
        # Keep track of dC/dX for each model where C is the cross entropy function
        dCdX = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        # Keep track of dF/dX for each model where F, is the Carlini-Wagner loss function (for updating alpha)
        dFdX = torch.zeros(numSamples, xShape[0], xShape[1],
                           xShape[2])  # Change to the math here to take in account all objecitve functions
        # dFdX = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        # Go through each model and compute dC/dX
        for m in range(0, len(modelListPlus)):
            # dataLoaderCurrent = modelListPlus[m].formatDataLoader(dataLoaderCurrent)
            dCdXTemp = FGSMNativeGradient(device, dataLoaderCurrent, modelListPlus[m])
            if modelListPlus[m].modelName.find("ViT") >= 0:
                attmap = GetAttention(dataLoaderCurrent, modelListPlus[m])
                attmap = torch.nn.functional.interpolate(attmap, size=(xShape[1], xShape[2]))
                dCdX[m] = dCdX[m] * attmap
            # Resize the graident to be the correct size and save it
            dCdX[m] = torch.nn.functional.interpolate(dCdXTemp, size=(xShape[1], xShape[2]))
            # Now compute the inital adversarial example with the base alpha
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            xGradientCumulative = xGradientCumulative + alpha[m] * dCdX[m]
        xAdvStepOne = xAdv + epsStep * xGradientCumulative.sign()
        xAdvStepOne = torch.min(xOriMax, xAdvStepOne)
        xAdvStepOne = torch.max(xOriMin, xAdvStepOne)
        # Do the clipping
        xAdvStepOne = torch.clamp(xAdvStepOne, clipMin, clipMax)
        # Convert the current xAdv to dataloader
        dataLoaderStepOne = DMP.TensorToDataLoader(xAdvStepOne, yClean, transforms=None,
                                                   batchSize=dataLoader.batch_size, randomizer=None)

        print("===Pre-Alpha Optimization===")
        costMultiplier = torch.zeros(len(modelListPlus), numSamples)
        # costCumulative = torch.zeros(numSamples)
        for m in range(0, len(modelListPlus)):
            cost = CheckCarliniLoss(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses)
            costMultiplier[m] = CarliniSingleSampleLoss(device, dataLoaderStepOne, modelListPlus[m], confidence,
                                                        nClasses)
            print("For model", m, "the Carlini value is", cost)
        # Now have to compute fraction for each sample
        # for n in range(0, numSamples):
        #    for m in range(0, len(modelListPlus)):
        #        costCumulative[n] = costCumulative[n] + costMultiplier[m, n]
        ##Now do the division
        # for n in range(0, numSamples):
        #    costMultiplier[m,n] = costMultiplier[m,n]/costCumulative[n]

        # Compute dF/dX (cumulative)
        for m in range(0, len(modelListPlus)):
            dFdX = dFdX + torch.nn.functional.interpolate(
                dFdXCompute(device, dataLoaderStepOne, modelListPlus[m], confidence, nClasses),
                size=(xShape[1], xShape[2]))
        # Compute dX/dAlpha
        dXdAlpha = dXdAlphaCompute(fittingFactor, epsStep, alpha, dCdX, len(modelListPlus), numSamples, xShape)
        # Compute dF/dAlpha = dF/dx * dX/dAlpha
        dFdAlpha = torch.zeros(len(modelListPlus), numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            # dFdAlpha = dFdX * dXdAlpha[m]
            dFdAlpha[m] = dFdX * dXdAlpha[m]
        # Now time to update alpha
        alpha = alpha - dFdAlpha * alphaLearningRate
        # for m in range(0, len(modelListPlus)):
        #    for n in range(0, numSamples):
        ##        print(alpha.shape)
        ##        print(dFdAlpha.shape)
        ##        print(costMultiplier.shape)
        #        alpha[m,n] = alpha[m,n] - dFdAlpha[m,n]*costMultiplier[m,n]*alphaLearningRate

        # Compute final adversarial example using best alpha
        xGradientCumulativeB = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            xGradientCumulativeB = xGradientCumulativeB + alpha[m] * dCdX[m]
        #     if modelListPlus[m].modelName.find("ViT") >= 0:
        #         attmap = GetAttention(dataLoaderCurrent, modelListPlus[m])
        #         attmap = torch.nn.functional.interpolate(attmap, size=(xShape[1], xShape[2]))
        #         xGradientCumulativeB = xGradientCumulativeB + alpha[m] * dCdX[m] * attmap
        #     else:
        #         xGradientCumulativeB = xGradientCumulativeB + alpha[m] * dCdX[m]

        # for m in range(0, len(modelListPlus)):
        #    for n in range(0, numSamples):
        #        xGradientCumulativeB[n] = xGradientCumulativeB[n] + dCdX[m,n]*costMultiplier[m,n]
        #    #xGradientCumulativeB = xGradientCumulativeB + alpha[m]*dCdX[m]*costMultiplier[m]
        xAdv = xAdv + epsStep * xGradientCumulativeB.sign()
        xAdv = torch.min(xOriMax, xAdv)
        xAdv = torch.max(xOriMin, xAdv)
        # clipMin = 0.0
        # clipMax = 1.0
        xAdv = torch.clamp(xAdv, clipMin, clipMax)
        # Convert the current xAdv to dataloader
        dataLoaderCurrent = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,
                                                   randomizer=None)

        # Debug HERE
        print("===Post-Alpha Optimization===")
        for m in range(0, len(modelListPlus)):
            cost = CheckCarliniLoss(device, dataLoaderCurrent, modelListPlus[m], confidence, nClasses)
            print("For model", m, "the Carlini value is", cost)

        if i == 20 or i == 40 or i == 60:
            accArrayProtoSAGA = torch.zeros(
                numSamples).to(device)  # Create an array with one entry for ever sample in the dataset
            for idx in range(0, len(modelListPlus)):
                accArray = modelListPlus[idx].validateDA(dataLoaderCurrent)
                accArrayProtoSAGA = accArrayProtoSAGA + accArray
                print("ProtoSAGA Acc " + modelListPlus[idx].modelName + ":", accArray.sum() / numSamples)
            MV_ProtoSAGA_acc = (accArrayProtoSAGA == 0).sum() / numSamples
            print('ProtoSAGA attack successful rate: ', MV_ProtoSAGA_acc.data.cpu().numpy())
            # if i == 20:
            #     alphaLearningRate = 10000
            #     print('alphaLearningRate set to ', alphaLearningRate)
            # if i == 40:
            #     alphaLearningRate = 1000
            #     print('alphaLearningRate set to ', alphaLearningRate)

    identifier = 'SuperSAGE_' + time.ctime(time.time()) + '.pt'
    torch.save((xAdv, yClean), identifier)
    print('Save the adv set ', identifier)
    identifier = 'Clean_' + time.ctime(time.time()) + '.pt'
    torch.save((xClean, yClean), identifier)
    print('Save the clean set ', identifier)
    return dataLoaderCurrent


def GetAttention(dLoader, modelPlus):
    dLoader = modelPlus.formatDataLoader(dLoader)
    numSamples = len(dLoader.dataset)
    attentionMaps = torch.zeros(numSamples, modelPlus.imgSizeH, modelPlus.imgSizeW, 3)
    currentIndexer = 0
    model = modelPlus.model.to(modelPlus.device)
    for ii, (x, y) in enumerate(dLoader):
        x = x.to(modelPlus.device)
        y = y.to(modelPlus.device)
        bsize = x.size()[0]
        attentionMapBatch = get_attention_map(model, x, bsize)
        # for i in range(0, dLoader.batch_size):
        for i in range(0, bsize):
            attentionMaps[currentIndexer] = attentionMapBatch[i]
            currentIndexer = currentIndexer + 1
    del model
    torch.cuda.empty_cache()
    print("attention maps generated")
    # change order
    attentionMaps = attentionMaps.permute(0, 3, 1, 2)
    return attentionMaps


def get_attention_map(model, xbatch, batch_size, img_size=224):
    attentionMaps = torch.zeros(batch_size, img_size, img_size, 3)
    index = 0
    for i in range(0, batch_size):
        ximg = xbatch[i].cpu().numpy().reshape(1, 3, img_size, img_size)
        ximg = torch.tensor(ximg).cuda()
        model.eval()
        res, att_mat = model.forward2(ximg)
        # model should return attention_list for all attention_heads
        # each element in the list contains attention for each layer

        att_mat = torch.stack(att_mat).squeeze(1)
        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat.cpu().detach() + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), (img_size, img_size))[..., np.newaxis]
        mask = np.concatenate((mask,) * 3, axis=-1)
        # print(mask.shape)
        attentionMaps[index] = torch.from_numpy(mask)
        index = index + 1
    return attentionMaps


# Compute dX/dAlpha for each model m
def dXdAlphaCompute(fittingFactor, epsStep, alpha, dCdX, numModels, numSamples, xShape):
    # Allocate memory for the solution
    dXdAlpha = torch.zeros(numModels, numSamples, xShape[0], xShape[1], xShape[2])
    innerSum = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    # First compute the inner summation sum m=1,...M: a_{m}*dC/dX_{m}
    for m in range(0, numModels):
        innerSum = innerSum + alpha[m] * dCdX[m]
    # Multiply inner sum by the fitting factor to approximate the sign(.) function
    innerSum = innerSum * fittingFactor
    # Now compute the sech^2 of the inner sum
    innerSumSecSquare = SechSquared(innerSum)
    # Now do the final computation to get dX/dAlpha (may not actually need for loop)
    for m in range(0, numModels):
        dXdAlpha[m] = fittingFactor * epsStep * dCdX[m] * innerSumSecSquare
    # All done so return
    return dXdAlpha


# Compute sech^2(x) using torch functions
def SechSquared(x):
    y = 4 * torch.exp(2 * x) / ((torch.exp(2 * x) + 1) * (torch.exp(2 * x) + 1))
    return y


# Custom loss function for updating alpha
def UntargetedCarliniLoss(logits, targets, confidence, nClasses, device):
    # This converts the normal target labels to one hot vectors e.g. y=1 will become [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    yOnehot = torch.nn.functional.one_hot(targets, nClasses).to(torch.float)
    zC = torch.max(yOnehot * logits,
                   1).values  # Need to use .values to get the Tensor because PyTorch max function doesn't want to give us a tensor
    zOther = torch.max((1 - yOnehot) * logits, 1).values
    loss = torch.max(zC - zOther + confidence, torch.tensor(0.0).to(device))
    return loss


# Native (no attack library) implementation of the FGSM attack in Pytorch
def FGSMNativeGradient(device, dataLoader, modelPlus):
    # Basic variable setup
    model = modelPlus.model
    model.eval()  # Change model to evaluation mode for the attack
    model.to(device)
    sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
    # Generate variables for storing the adversarial examples
    numSamples = len(sizeCorrectedLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(sizeCorrectedLoader)  # Get the shape of the input (there may be easier way to do this)
    xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  # just do dummy initalization, will be filled in later
    # Go through each sample
    tracker = 0
    for xData, yData in sizeCorrectedLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        # print("Processing up to sample=", tracker)
        # Put the data from the batch onto the device
        # xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        xDataTemp = xData.detach().to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        xDataTemp.requires_grad = True
        if modelPlus.modelName == 'SNN ResNet Backprop':
            functional.reset_net(model)  # Line to reset model memory to accodomate Spiking Jelly (new attack iteration)
            # Forward pass the data through the model
            output = model(xDataTemp).mean(0)
        else:
            # Forward pass the data through the model
            output = model(xDataTemp)
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = loss(output, yData).to(device)
        cost.backward()
        if modelPlus.modelName == 'SNN VGG-16 Backprop':
            xDataTempGrad = xDataTemp.grad.data.sum(-1)
        else:
            xDataTempGrad = xDataTemp.grad.data
        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xGradient[advSampleIndex] = xDataTempGrad[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
        # Not sure if we need this but do some memory clean up
        del xDataTemp
        torch.cuda.empty_cache()
    # Memory management
    del model
    torch.cuda.empty_cache()
    return xGradient


# Do the computation from scratch to get the correctly identified overlapping examples
# Note these samples will be the same size as the input size required by the 0th model
def GetFirstCorrectlyOverlappingSamplesBalanced(device, sampleNum, numClasses, dataLoader, modelPlusList):
    numModels = len(modelPlusList)
    totalSampleNum = len(dataLoader.dataset)
    # First check if modelA needs resize
    xTestOrig, yTestOrig = DMP.DataLoaderToTensor(dataLoader)
    # We need to resize first
    if modelPlusList[0].imgSizeH != xTestOrig.shape[2] or modelPlusList[0].imgSizeW != xTestOrig.shape[3]:
        xTestOrigResize = torch.zeros(xTestOrig.shape[0], 3, modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)
        rs = torchvision.transforms.Resize(
            (modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW))  # resize the samples for model A
        # Go through every sample
        for i in range(0, xTestOrig.shape[0]):
            xTestOrigResize[i] = rs(xTestOrig[i])  # resize to match dimensions required by modelA
        # Make a new dataloader
        dataLoader = DMP.TensorToDataLoader(xTestOrigResize, yTestOrig, transforms=None,
                                            batchSize=dataLoader.batch_size, randomizer=None)
    # Get accuracy array for each model
    accArrayCumulative = torch.zeros(totalSampleNum)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(dataLoader)
        accArrayCumulative = accArrayCumulative + accArray
    # Do some basic error checking
    if sampleNum % numClasses != 0:
        raise ValueError("Number of samples not divisable by the number of classes")
    # Basic variable setup
    samplePerClassCount = torch.zeros(numClasses)  # keep track of samples per class
    maxRequireSamplesPerClass = int(sampleNum / numClasses)  # Max number of samples we need per class
    xTest, yTest = DMP.DataLoaderToTensor(dataLoader)  # Get all the data as tensors
    # Memory for the solution
    xClean = torch.zeros(sampleNum, 3, modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)
    yClean = torch.zeros(sampleNum)
    sampleIndexer = 0
    # Go through all the samples
    for i in range(0, totalSampleNum):
        currentClass = int(yTest[i])
        # Check to make sure all classifiers identify the sample correctly AND we don't have enough of this class yet
        if accArrayCumulative[i] == numModels and samplePerClassCount[currentClass] < maxRequireSamplesPerClass:
            # xClean[sampleIndexer] = rs(xTest[i]) #resize to match dimensions required by modelA
            xClean[sampleIndexer] = xTest[i]
            yClean[sampleIndexer] = yTest[i]
            sampleIndexer = sampleIndexer + 1  # update the indexer
            samplePerClassCount[currentClass] = samplePerClassCount[
                                                    currentClass] + 1  # Update the number of samples for this class
    # Check the over all number of samples as well
    if sampleIndexer != sampleNum:
        print("Not enough clean samples found.")
    # Do some error checking on the classes
    for i in range(0, numClasses):
        if samplePerClassCount[i] != maxRequireSamplesPerClass:
            print(samplePerClassCount[i])
            raise ValueError("We didn't find enough of class: " + str(i))
    # Conver the solution into a dataloader
    cleanDataLoader = DMP.TensorToDataLoader(xClean, yClean, transforms=None, batchSize=modelPlusList[0].batchSize,
                                             randomizer=None)
    # Do one last check to make sure all samples identify the clean loader correctly
    for i in range(0, numModels):
        cleanAcc = modelPlusList[i].validateD(cleanDataLoader)
        if cleanAcc != 1.0:
            print("Clean Acc " + modelPlusList[i].modelName + ":", cleanAcc)
            raise ValueError("The clean accuracy is not 1.0")
    # All error checking done, return the clean balanced loader
    return cleanDataLoader


def dFdXCompute(device, dataLoader, modelPlus, confidence, nClasses):
    # Basic variable setup
    model = modelPlus.model
    model.eval()  # Change model to evaluation mode for the attack
    model.to(device)
    sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
    # Generate variables for storing the adversarial examples
    numSamples = len(sizeCorrectedLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(sizeCorrectedLoader)  # Get the shape of the input (there may be easier way to do this)
    xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  # just do dummy initalization, will be filled in later
    # Go through each sample
    tracker = 0
    for xData, yData in sizeCorrectedLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        # print("Processing up to sample=", tracker)
        # Put the data from the batch onto the device
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        xDataTemp.requires_grad = True
        if modelPlus.modelName == 'SNN ResNet Backprop':
            functional.reset_net(model)  # Line to reset model memory to accodomate Spiking Jelly (new attack iteration)
            # Forward pass the data through the model
            outputLogits = model(xDataTemp).mean(0)
        else:
            # Forward pass the data through the model
            outputLogits = model(xDataTemp)
        # Calculate the loss with respect to the Carlini Wagner loss function
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = UntargetedCarliniLoss(outputLogits, yData, confidence, nClasses, device).sum().to(
            device)  # Not sure about the sum
        cost.backward()
        if modelPlus.modelName == 'SNN VGG-16 Backprop':
            xDataTempGrad = xDataTemp.grad.data.sum(-1)
        else:
            xDataTempGrad = xDataTemp.grad.data
        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xGradient[advSampleIndex] = xDataTempGrad[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
        # Not sure if we need this but do some memory clean up
        del xDataTemp
        torch.cuda.empty_cache()
    # Memory management
    del model
    torch.cuda.empty_cache()
    return xGradient


def CheckCarliniLoss(device, dataLoader, modelPlus, confidence, nClasses):
    # Basic variable setup
    model = modelPlus.model
    model.eval()  # Change model to evaluation mode for the attack
    model.to(device)
    sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
    # Generate variables for storing the adversarial examples
    numSamples = len(sizeCorrectedLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(sizeCorrectedLoader)  # Get the shape of the input (there may be easier way to do this)
    xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  # just do dummy initalization, will be filled in later
    # Go through each sample
    tracker = 0
    cumulativeCost = 0
    for xData, yData in sizeCorrectedLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        # print("Processing up to sample=", tracker)
        # Put the data from the batch onto the device
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Forward pass the data through the model
        if modelPlus.modelName == 'SNN ResNet Backprop':
            functional.reset_net(model)  # Line to reset model memory to accodomate Spiking Jelly (new attack iteration)
            # Forward pass the data through the model
            outputLogits = model(xDataTemp).mean(0)
        else:
            # Forward pass the data through the model
            outputLogits = model(xDataTemp)
        # Calculate the loss with respect to the Carlini Wagner loss function
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = UntargetedCarliniLoss(outputLogits, yData, confidence, nClasses, device).sum()  # Not sure about the sum
        cumulativeCost = cumulativeCost + cost.to("cpu")
        cost.backward()
        # Not sure if we need this but do some memory clean up
        del xDataTemp
        torch.cuda.empty_cache()
    # Memory management
    del model
    torch.cuda.empty_cache()
    return cumulativeCost


# Get the loss associated with single samples
def CarliniSingleSampleLoss(device, dataLoader, modelPlus, confidence, nClasses):
    # Basic variable setup
    model = modelPlus.model
    model.eval()  # Change model to evaluation mode for the attack
    model.to(device)
    sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
    # Generate variables for storing the adversarial examples
    numSamples = len(sizeCorrectedLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(sizeCorrectedLoader)  # Get the shape of the input (there may be easier way to do this)
    xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    # Variables to store the associated costs values
    costValues = torch.zeros(numSamples)
    batchSize = 0  # just do dummy initalization, will be filled in later
    # Go through each sample
    tracker = 0
    for xData, yData in sizeCorrectedLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        # Put the data from the batch onto the device
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        if modelPlus.modelName == 'SNN ResNet Backprop':
            functional.reset_net(model)  # Line to reset model memory to accodomate Spiking Jelly (new attack iteration)
            # Forward pass the data through the model
            outputLogits = model(xDataTemp).mean(0)
        else:
            # Forward pass the data through the model
            outputLogits = model(xDataTemp)
        # Calculate the loss with respect to the Carlini Wagner loss function
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = UntargetedCarliniLoss(outputLogits, yData, confidence, nClasses, device)
        cost.sum().backward()
        # Store the current cost values
        costValues[tracker:tracker + batchSize] = cost.to("cpu")
        tracker = tracker + batchSize
        # Not sure if we need this but do some memory clean up
        del xDataTemp
        torch.cuda.empty_cache()
    # Memory management
    del model
    torch.cuda.empty_cache()
    return costValues


# Do the computation from scratch to get the correctly identified overlapping examples
# Note these samples will be the same size as the input size required by the 0th model
def GetFirstCorrectlyOverlappingSamplesBalanced(device, sampleNum, numClasses, dataLoader, modelPlusList):
    numModels = len(modelPlusList)
    totalSampleNum = len(dataLoader.dataset)
    # # First check if modelA needs resize
    # xTestOrig, yTestOrig = DMP.DataLoaderToTensor(dataLoader)
    # # # We need to resize first
    # # if modelPlusList[1].imgSizeH != xTestOrig.shape[2] or modelPlusList[1].imgSizeW != xTestOrig.shape[3]:
    # #     xTestOrigResize = torch.zeros(xTestOrig.shape[0], 3, modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)
    # #     rs = torchvision.transforms.Resize(
    # #         (modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW))  # resize the samples for model A
    # #     # Go through every sample
    # #     for i in range(0, xTestOrig.shape[0]):
    # #         xTestOrigResize[i] = rs(xTestOrig[i])  # resize to match dimensions required by modelA
    # #     # Make a new dataloader
    # #     dataLoader = DMP.TensorToDataLoader(xTestOrigResize, yTestOrig, transforms=None,
    # #                                         batchSize=dataLoader.batch_size, randomizer=None)
    # Get accuracy array for each model
    accArrayCumulative = torch.zeros(totalSampleNum).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        path = "./imgnet_accArray_imgnet_" + modelPlusList[i].modelName
        if os.path.exists(path):
            accArray = torch.load(path)
        else:
            accArray = modelPlusList[i].validateDA(dataLoader)
            torch.save(accArray, "./imgnet_accArray_imgnet_"+modelPlusList[i].modelName)
        accArrayCumulative = accArrayCumulative + accArray
    # Do some basic error checking
    if sampleNum % numClasses != 0:
        raise ValueError("Number of samples not divisable by the number of classes")
    # Basic variable setup
    samplePerClassCount = torch.zeros(numClasses)  # keep track of samples per class
    maxRequireSamplesPerClass = int(sampleNum / numClasses)  # Max number of samples we need per class
    xTest, yTest = DMP.DataLoaderToTensor(dataLoader)  # Get all the data as tensors
    # Memory for the solution
    xClean = torch.zeros(sampleNum, 3, modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)
    yClean = torch.zeros(sampleNum)
    sampleIndexer = 0
    # Go through all the samples
    for i in range(0, totalSampleNum):
        currentClass = int(yTest[i])
        # Check to make sure all classifiers identify the sample correctly AND we don't have enough of this class yet
        if accArrayCumulative[i] == numModels and samplePerClassCount[currentClass] < maxRequireSamplesPerClass:
            # xClean[sampleIndexer] = rs(xTest[i]) #resize to match dimensions required by modelA
            xClean[sampleIndexer] = xTest[i]
            yClean[sampleIndexer] = yTest[i]
            sampleIndexer = sampleIndexer + 1  # update the indexer
            samplePerClassCount[currentClass] = samplePerClassCount[
                                                    currentClass] + 1  # Update the number of samples for this class
    # Check the over all number of samples as well
    if sampleIndexer != sampleNum:
        print("Not enough clean samples found.")
    # Do some error checking on the classes
    for i in range(0, numClasses):
        if samplePerClassCount[i] != maxRequireSamplesPerClass:
            print(samplePerClassCount[i])
            # raise ValueError("We didn't find enough of class: " + str(i))
            print("We didn't find enough of class: " + str(i))
    # Conver the solution into a dataloader
    cleanDataLoader = DMP.TensorToDataLoader(xClean, yClean, transforms=None, batchSize=modelPlusList[0].batchSize,
                                             randomizer=None)
    # Do one last check to make sure all samples identify the clean loader correctly
    for i in range(0, numModels):
        cleanAcc = modelPlusList[i].validateD(cleanDataLoader)
        if cleanAcc != 1.0:
            print("Clean Acc " + modelPlusList[i].modelName + ":", cleanAcc)
            # raise ValueError("The clean accuracy is not 1.0")
    # All error checking done, return the clean balanced loader
    return cleanDataLoader




# Do the computation from scratch to get the correctly identified overlapping examples
# Note these samples will be the same size as the input size required by the 0th model
def GetFirstCorrectlyOverlappingSamplesBalanced_twosize(device, sampleNum, numClasses, dataLoader, dataloader2, modelPlusList):
    numModels = len(modelPlusList)
    totalSampleNum = len(dataLoader.dataset)
    # # First check if modelA needs resize
    # xTestOrig, yTestOrig = DMP.DataLoaderToTensor(dataLoader)
    # # # We need to resize first
    # # if modelPlusList[1].imgSizeH != xTestOrig.shape[2] or modelPlusList[1].imgSizeW != xTestOrig.shape[3]:
    # #     xTestOrigResize = torch.zeros(xTestOrig.shape[0], 3, modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)
    # #     rs = torchvision.transforms.Resize(
    # #         (modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW))  # resize the samples for model A
    # #     # Go through every sample
    # #     for i in range(0, xTestOrig.shape[0]):
    # #         xTestOrigResize[i] = rs(xTestOrig[i])  # resize to match dimensions required by modelA
    # #     # Make a new dataloader
    # #     dataLoader = DMP.TensorToDataLoader(xTestOrigResize, yTestOrig, transforms=None,
    # #                                         batchSize=dataLoader.batch_size, randomizer=None)
    # Get accuracy array for each model
    accArrayCumulative = torch.zeros(totalSampleNum).to(device)  # Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        path = "./imgnet_accArray_imgnet_" + modelPlusList[i].modelName
        if os.path.exists(path):
            accArray = torch.load(path)
        else:
            accArray = modelPlusList[i].validateDA(dataloader2)
            torch.save(accArray, "./imgnet_accArray_imgnet_"+modelPlusList[i].modelName)
        accArrayCumulative = accArrayCumulative + accArray
    # Do some basic error checking
    if sampleNum % numClasses != 0:
        raise ValueError("Number of samples not divisable by the number of classes")
    # Basic variable setup
    samplePerClassCount = torch.zeros(numClasses)  # keep track of samples per class
    maxRequireSamplesPerClass = int(sampleNum / numClasses)  # Max number of samples we need per class
    xTest, yTest = DMP.DataLoaderToTensor(dataLoader)  # Get all the data as tensors
    # Memory for the solution
    xClean = torch.zeros(sampleNum, 3, modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)
    yClean = torch.zeros(sampleNum)
    sampleIndexer = 0
    # Go through all the samples
    for i in range(0, totalSampleNum):
        currentClass = int(yTest[i])
        # Check to make sure all classifiers identify the sample correctly AND we don't have enough of this class yet
        if accArrayCumulative[i] == numModels and samplePerClassCount[currentClass] < maxRequireSamplesPerClass:
            # xClean[sampleIndexer] = rs(xTest[i]) #resize to match dimensions required by modelA
            xClean[sampleIndexer] = xTest[i]
            yClean[sampleIndexer] = yTest[i]
            sampleIndexer = sampleIndexer + 1  # update the indexer
            samplePerClassCount[currentClass] = samplePerClassCount[
                                                    currentClass] + 1  # Update the number of samples for this class
    # Check the over all number of samples as well
    if sampleIndexer != sampleNum:
        print("Not enough clean samples found.")
    # Do some error checking on the classes
    for i in range(0, numClasses):
        if samplePerClassCount[i] != maxRequireSamplesPerClass:
            print(samplePerClassCount[i])
            # raise ValueError("We didn't find enough of class: " + str(i))
            print("We didn't find enough of class: " + str(i))
    # Conver the solution into a dataloader
    cleanDataLoader = DMP.TensorToDataLoader(xClean, yClean, transforms=None, batchSize=modelPlusList[0].batchSize,
                                             randomizer=None)
    torch.save(cleanDataLoader,
               "./imgnet_cleanLoader_imgnet_" + modelPlusList[0].modelName + "_" + modelPlusList[1].modelName)

    # Do one last check to make sure all samples identify the clean loader correctly
    for i in range(0, numModels):
        cleanAcc = modelPlusList[i].validateD(cleanDataLoader)
        if cleanAcc != 1.0:
            print("Clean Acc " + modelPlusList[i].modelName + ":", cleanAcc)
            # raise ValueError("The clean accuracy is not 1.0")
    # All error checking done, return the clean balanced loader
    return cleanDataLoader

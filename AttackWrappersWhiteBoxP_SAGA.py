#Attack wrappers class for FGSM and MIM (no extra library implementation) to be used in conjunction with 
#the adaptive black-box attack 
import torch 
import DataManagerPytorch as DMP
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from spikingjelly.clock_driven import functional
import time

#Native (no attack library) implementation of the FGSM attack in Pytorch 
def FGSMNativePytorch(device, dataLoader, model, epsilonMax, clipMin, clipMax, mean, std, targeted):
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    #Go through each sample 
    tracker = 0
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        #print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        # xDataTemp = xData.to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        xDataTemp.requires_grad = True
        # Forward pass the data through the model
        output = model(xDataTemp)
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = loss(output, yData).to(device)
        cost.backward()
        # Collect datagrad
        #xDataGrad = xDataTemp.grad.data
        ###Here we actual compute the adversarial sample 
        # Collect the element-wise sign of the data gradient
        signDataGrad = xDataTemp.grad.data.sign()
        # print('signDataGrad: ', signDataGrad.unique())
        # Create the perturbed image by adjusting each pixel of the input image
        #print("xData:", xData.is_cuda)
        #print("SignGrad:", signDataGrad.is_cuda)
        if targeted == True:
            perturbedImage = (xDataTemp * std + mean) - epsilonMax*signDataGrad #Go negative of gradient
        else:
            perturbedImage = (xDataTemp * std + mean) + epsilonMax*signDataGrad
        # Adding clipping to maintain the range
        perturbedImage = torch.clamp(perturbedImage, clipMin, clipMax)
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = (perturbedImage[j] - mean)/std
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
        #Not sure if we need this but do some memory clean up 
        del xDataTemp
        del signDataGrad
        torch.cuda.empty_cache()
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader


#Native (no attack library) implementation of the FGSM attack in Pytorch
def FGSMModifyPytorch(device, dataLoader, model, epsilonMax, clipMin, clipMax, targeted):
    model.eval() #Change model to evaluation mode for the attack
    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0 #just do dummy initalization, will be filled in later
    #Go through each sample
    tracker = 0
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        # print('device: ', device)
        #Put the data from the batch onto the device
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        # xDataTemp = xData.to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        xDataTemp.requires_grad = True
        # Forward pass the data through the model
        output = model(xDataTemp)
        # print('after inference')
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = loss(output, yData).to(device)
        cost.backward()
        # print('calculate gradients')
        # Collect datagrad
        #xDataGrad = xDataTemp.grad.data
        ###Here we actual compute the adversarial sample
        # Collect the element-wise sign of the data gradient
        signDataGrad = xDataTemp.grad.data.sign()
        # print('signDataGrad: ', signDataGrad.unique())
        # Create the perturbed image by adjusting each pixel of the input image
        # print("xData:", xData.is_cuda)
        # print("SignGrad:", signDataGrad.is_cuda)
        if targeted == True:
            perturbedImage = xData - epsilonMax*signDataGrad.cpu().detach() #Go negative of gradient
        else:
            perturbedImage = xData + epsilonMax*signDataGrad.cpu().detach()
        # Adding clipping to maintain the range
        perturbedImage = torch.clamp(perturbedImage, clipMin, clipMax)
        # print('perturbedImage: ', perturbedImage)
        #Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = perturbedImage[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
        #Not sure if we need this but do some memory clean up
        del xDataTemp
        del signDataGrad
        # torch.cuda.empty_cache()
    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

#Native (no attack library) implementation of the MIM attack in Pytorch
#This is only for the L-infinty norm and cross entropy loss function
def MIMNativePytorch_cnn(device, dataLoader, model, decayFactor, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, mean, std, targeted):
    model.eval() #Change model to evaluation mode for the attack
    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0 #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        xOridata = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        #Initalize memory for the gradient momentum
        gMomentum = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2])
        #Do the attack for a number of steps
        for attackStep in range(0, numSteps):
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            #Update momentum term
            gMomentum = decayFactor*gMomentum + GradientNormalizedByL1(xAdvCurrent.grad)
            #Update the adversarial sample
            if targeted == True:
                advTemp = (xAdvCurrent * std + mean) - (epsilonStep*torch.sign(gMomentum)).to(device)
            else:
                advTemp = (xAdvCurrent * std + mean) + (epsilonStep*torch.sign(gMomentum)).to(device)
            #Adding clipping to maintain the range
            # delta = torch.clamp(advTemp - (xOridata * std + mean), min=-epsilonMax, max=epsilonMax)
            # xAdvCurrent = torch.clamp((xOridata * std + mean) + delta, min=clipMin, max=clipMax).detach_()
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
            xAdvCurrent = (xAdvCurrent - mean) / std
        #Save the adversarial images from the batch
        for j in range(0, batchSize):
            # xAdv[advSampleIndex] = (xAdvCurrent[j] - mean) / std
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader



#Native (no attack library) implementation of the MIM attack in Pytorch 
#This is only for the L-infinty norm and cross entropy loss function 
def MIMNativePytorch(device, dataLoader, model, modelPlus, decayFactor, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, targeted):
    model.eval() #Change model to evaluation mode for the attack
    dataLoader = modelPlus.formatDataLoader(dataLoader)
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample 
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xAdvCurrent = xData.to(device)
        xOridata = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        #Initalize memory for the gradient momentum
        gMomentum = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2])
        #Do the attack for a number of steps
        for attackStep in range(0, numSteps):   
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            #Update momentum term 
            gMomentum = decayFactor*gMomentum + GradientNormalizedByL1(xAdvCurrent.grad)
            #Update the adversarial sample 
            if targeted == True:
                advTemp = (xAdvCurrent ) - (epsilonStep*torch.sign(gMomentum)).to(device)
            else:
                advTemp = (xAdvCurrent ) + (epsilonStep*torch.sign(gMomentum)).to(device)
            #Adding clipping to maintain the range
            # delta = torch.clamp(advTemp - (xOridata * std + mean), min=-epsilonMax, max=epsilonMax)
            # xAdvCurrent = torch.clamp((xOridata * std + mean) + delta, min=clipMin, max=clipMax).detach_()
            advTemp = torch.min(xOridata + epsilonMax, advTemp)
            advTemp = torch.max(xOridata - epsilonMax, advTemp)
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()

        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            # xAdv[advSampleIndex] = (xAdvCurrent[j] - mean) / std
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
    # torch.save((xAdv, yClean), 'MIM_CIFAR10.pt')
    identifier = 'MIM_' + time.ctime(time.time()) + '.pt'
    torch.save((xAdv, yClean), identifier)
    print('Save the MIM_adv set ', identifier)
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader


def PGDNativePytorch(device, dataLoader, model, modelPlus, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, targeted):
    model.eval() #Change model to evaluation mode for the attack
    dataLoader = modelPlus.formatDataLoader(dataLoader)
    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0 #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device
        xAdvCurrent = xData.to(device).detach()
        xOridata = xData.to(device).detach()
        yCurrent = yData.type(torch.LongTensor).to(device)
        xOriMax = (xOridata) + epsilonMax
        xOriMin = (xOridata) - epsilonMax
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        #Initalize memory for the gradient momentum
        #gMomentum = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2])
        #Do the attack for a number of steps
        for attackStep in range(0, numSteps):
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            #Update momentum term
            #gMomentum = decayFactor*gMomentum + GradientNormalizedByL1(xAdvCurrent.grad)
            #Update the adversarial sample
            if targeted == True:
                advTemp = (xAdvCurrent) - (epsilonStep*torch.sign(xAdvCurrent.grad)).to(device)
            else:
                advTemp = (xAdvCurrent) + (epsilonStep*torch.sign(xAdvCurrent.grad)).to(device)
            #Adding clipping to maintain the range
            advTemp = torch.min(xOriMax, advTemp)
            advTemp = torch.max(xOriMin, advTemp)
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()

        #Save the adversarial images from the batch
        for j in range(0, batchSize):
            # xAdv[advSampleIndex] = (xAdvCurrent[j] - mean) / std
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
    # torch.save((xAdv, yClean), 'PGD_CIFAR10.pt')
    identifier = 'PGD_' + time.ctime(time.time()) + '.pt'
    torch.save((xAdv, yClean), identifier)
    print('Save the PGD_adv set ', identifier)
    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader





# Main attack method, takes in a list of models and a clean data loader
# Returns a dataloader with the adverarial samples and corresponding clean labels
def SelfAttentionGradientAttack(device, epsMax, numSteps, epsStep, modelListPlus, coefficientArray, dataLoader, clipMin,
                                clipMax, mean, std):
    # Basic graident variable setup
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    xAdv = xClean.detach()  # Set the initial adversarial samples
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    xOridata = xClean.detach()
    print('xClean: ', xClean.shape)
    xOriMax = xOridata + epsMax
    xOriMin = xOridata - epsMax
    # Compute eps step
    # epsStep = epsMax / numSteps
    dataLoaderCurrent = dataLoader
    for i in range(0, numSteps):
        print("Running Step=", i)
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        for m in range(0, len(modelListPlus)):
            # currentModelPlus = modelListPlus[m]
            # First get the gradient from the model
            # dataLoaderCurrent = currentModelPlus.formatDataLoader(dataLoaderCurrent)
            xGradientCurrent = FGSMNativeGradient(device, dataLoaderCurrent, modelListPlus[m])
            # Resize the graident to be the correct size
            xGradientCurrent = torch.nn.functional.interpolate(xGradientCurrent, size=(xShape[1], xShape[2]))
            # Add the current computed gradient to the result
            if modelListPlus[m].modelName.find("ViT") >= 0:
                attmap = GetAttention(dataLoaderCurrent, modelListPlus[m])
                attmap = torch.nn.functional.interpolate(attmap, size=(xShape[1], xShape[2]))
                xGradientCumulative = xGradientCumulative + coefficientArray[m] * xGradientCurrent * attmap
            else:
                xGradientCumulative = xGradientCumulative + coefficientArray[m] * xGradientCurrent
        # Compute the sign of the graident and create the adversarial example
        # xAdv = xAdv + epsStep * xGradientCumulative.sign()
        xAdv = xAdv + epsStep * xGradientCumulative.sign()
        xAdv = torch.min(xOriMax, xAdv)
        xAdv = torch.max(xOriMin, xAdv)
        # Do the clipping
        xAdv = torch.clamp(xAdv, clipMin, clipMax)
        # Convert the result to dataloader
        dataLoaderCurrent = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,
                                                   randomizer=None)
    identifier = 'SAGE_' + time.ctime(time.time()) + '.pt'
    torch.save((xAdv, yClean), identifier)
    print('Save the adv set ', identifier)

    return dataLoaderCurrent


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
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
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


def GetAttention(dLoader, modelPlus):
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

#Native (no attack library) implementation of the MIM attack in Pytorch
#This is only for the L-infinty norm and cross entropy loss function
def PGDNativePytorch_cnn(device, dataLoader, model, epsilonMax, epsilonStep, numSteps, clipMin, clipMax,  mean, std, targeted):
    model.eval() #Change model to evaluation mode for the attack
    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0 #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        # print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        #Initalize memory for the gradient momentum
        #gMomentum = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2])
        #Do the attack for a number of steps
        for attackStep in range(0, numSteps):
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            #Update momentum term
            #gMomentum = decayFactor*gMomentum + GradientNormalizedByL1(xAdvCurrent.grad)
            #Update the adversarial sample
            if targeted == True:
                advTemp = (xAdvCurrent * std + mean) - (epsilonStep*torch.sign(xAdvCurrent.grad)).to(device)
            else:
                advTemp = (xAdvCurrent * std + mean) + (epsilonStep*torch.sign(xAdvCurrent.grad)).to(device)
            #Adding clipping to maintain the range
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
            xAdvCurrent = (xAdvCurrent - mean) / std
        #Save the adversarial images from the batch
        for j in range(0, batchSize):
            # xAdv[advSampleIndex] = (xAdvCurrent[j] - mean) / std
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

def PGDNativePytorch_iccv(device, dataLoader, model, epsilonMax, epsilonStep, numSteps, clipMin, clipMax,  mean, std, targeted):
    model.eval() #Change model to evaluation mode for the attack
    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0 #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        # print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        xOri = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        #Initalize memory for the gradient momentum
        #gMomentum = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2])
        #Do the attack for a number of steps
        for attackStep in range(0, numSteps):
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            #Update momentum term
            #gMomentum = decayFactor*gMomentum + GradientNormalizedByL1(xAdvCurrent.grad)
            #Update the adversarial sample
            if targeted == True:
                advTemp = (xAdvCurrent * std + mean) - (epsilonStep*torch.sign(xAdvCurrent.grad)).to(device)
            else:
                advTemp = (xAdvCurrent * std + mean) + (epsilonStep*torch.sign(xAdvCurrent.grad)).to(device)
            #Adding clipping to maintain the range
            advTemp = torch.max(xOri * std + mean - epsilonMax, advTemp)
            advTemp = torch.min(xOri * std + mean + epsilonMax, advTemp)
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
            xAdvCurrent = (xAdvCurrent - mean) / std
        #Save the adversarial images from the batch
        for j in range(0, batchSize):
            # xAdv[advSampleIndex] = (xAdvCurrent[j] - mean) / std
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader


#Native (no attack library) implementation of the MIM attack in Pytorch 
#This is only for the L-infinty norm and cross entropy loss function 
def MIMNativePytorch_snn(device, dataLoader, model, decayFactor, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, targeted):
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample 
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        #Initalize memory for the gradient momentum
        gMomentum = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2])
        #Do the attack for a number of steps
        for attackStep in range(0, numSteps):   
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            #Update momentum term 
            gMomentum = decayFactor*gMomentum + GradientNormalizedByL1(xAdvCurrent.grad)
            #Update the adversarial sample 
            if targeted == True:
                advTemp = xAdvCurrent - (epsilonStep*torch.sign(gMomentum)).to(device)
            else:
                advTemp = xAdvCurrent + (epsilonStep*torch.sign(gMomentum)).to(device)
            #Adding clipping to maintain the range
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            # xAdv[advSampleIndex] = (xAdvCurrent[j] - 0.5) / 0.5
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index 
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

def GradientNormalizedByL1(gradient):
    #Do some basic error checking first
    if gradient.shape[1] != 3:
        raise ValueError("Shape of gradient is not consistent with an RGB image.")
    #basic variable setup
    batchSize = gradient.shape[0]
    colorChannelNum = gradient.shape[1]
    imgRows = gradient.shape[2]
    imgCols = gradient.shape[3]
    gradientNormalized = torch.zeros(batchSize, colorChannelNum, imgRows, imgCols)
    #Compute the L1 gradient for each color channel and normalize by the gradient 
    #Go through each color channel and compute the L1 norm
    for i in range(0, batchSize):
        for c in range(0, colorChannelNum):
           norm = torch.linalg.norm(gradient[i,c], ord=1)
           gradientNormalized[i,c] = gradient[i,c]/norm #divide the color channel by the norm
    return gradientNormalized



# This operation can all be done in one line but for readability later
# the projection operation is done in multiple steps for l-inf norm
def ProjectionOperation(xAdv, xClean, epsilonMax):
    # First make sure that xAdv does not exceed the acceptable range in the positive direction
    xAdv = torch.min(xAdv, xClean + epsilonMax)
    # Second make sure that xAdv does not exceed the acceptable range in the negative direction
    xAdv = torch.max(xAdv, xClean - epsilonMax)
    return xAdv


# Function for computing the model gradient
def GetModelGradient(device, model, xK, yK):
    # Define the loss function
    loss = torch.nn.CrossEntropyLoss()
    xK.requires_grad = True
    # Pass the inputs through the model
    outputs = model(xK.to(device))
    model.zero_grad()
    # Compute the loss
    cost = loss(outputs, yK)
    cost.backward()
    xKGrad = xK.grad
    #Do GPU memory clean up (important)
    del xK
    del cost
    del outputs
    del loss
    return xKGrad


def ComputePList(pList, startIndex, decrement):
    # p(j+1) = p(j) + max( p(j) - p(j-1) -0.03, 0.06))
    nextP = pList[startIndex] + max(pList[startIndex] - pList[startIndex - 1] - decrement, 0.06)
    # Check for base case
    if nextP >= 1.0:
        return pList
    else:
        # Need to further recur
        pList.append(nextP)
        ComputePList(pList, startIndex + 1, decrement)


def ComputeCheckPoints(Niter, decrement):
    # First compute the pList based on the decrement amount
    pList = [0, 0.22]  # Starting pList based on AutoAttack paper
    ComputePList(pList, 1, decrement)
    # Second compute the checkpoints from the pList
    wList = []
    for i in range(0, len(pList)):
        wList.append(int(np.ceil(pList[i] * Niter)))
    # There may duplicates in the list due to rounding so finally we remove duplicates
    wListFinal = []
    for i in wList:
        if i not in wListFinal:
            wListFinal.append(i)
    # Return the final list
    return wListFinal


# Condition two checks if the objective function and step size previously changed
def CheckConditionTwo(f, eta, checkPointIndex, checkPoints):
    currentCheckPoint = checkPoints[checkPointIndex]
    previousCheckPoint = checkPoints[checkPointIndex - 1]  # Get the previous checkpoint
    if eta[previousCheckPoint] == eta[currentCheckPoint] and f[previousCheckPoint] == f[currentCheckPoint]:
        return True
    else:
        return False


# Condition one checks the summation of objective function
def CheckConditionOne(f, checkPointIndex, checkPoints, targeted):
    sum = 0
    currentCheckPoint = checkPoints[checkPointIndex]
    previousCheckPoint = checkPoints[checkPointIndex - 1]  # Get the previous checkpoint
    # See how many times the objective function was growing bigger
    for i in range(previousCheckPoint, currentCheckPoint):  # Goes from w_(j-1) to w_(j) - 1
        if f[i + 1] > f[i]:
            sum = sum + 1
    ratio = 0.75 * (currentCheckPoint - previousCheckPoint)
    # For untargeted attack we want the objective function to increase
    if targeted == False and sum < ratio:  # This is condition 1 from the Autoattack paper
        return True
    elif targeted == True and sum > ratio:  # This is my interpretation of how the targeted attack would work (not 100% sure)
        return True
    else:
        return False


# Native (no attack library) implementation of the AutoAttack attack in Pytorch
# This is only for the L-infinty norm and cross entropy loss function
# This implementaiton is very GPU memory intensive
def AutoAttackNativePytorch(device, dataLoader, model, modelPlus, epsilonMax, etaStart, numSteps, clipMin, clipMax, targeted):
    # Setup attack variables:
    decrement = 0.
    model.to(device)
    model.eval() #Change model to evaluation mode for the attack
    dataLoader = modelPlus.formatDataLoader(dataLoader)
    wList = ComputeCheckPoints(numSteps, decrement)  # Get the list of checkpoints based on the number of iterations
    alpha = 0.75  # Weighting factor for momentum
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0  # Indexing variable for saving the adversarial example
    batchSize = 0  # just do dummy initalization, will be filled in later
    tracker = 0
    # lossIndividual = torch.nn.CrossEntropyLoss(reduction='none')
    # Go through each batch and run the attack
    for xData, yData in dataLoader:
        # Initialize the AutoAttack variables
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize  # Update the tracking variable
        print(tracker, end="\r")
        yK = yData.type(torch.LongTensor).to(device)  # Correct class labels which don't change in the iterations
        eta = torch.zeros(numSteps + 1, batchSize)  # Keep track of the step size for each sample
        eta[0, :] = etaStart  # Initalize eta values as the starting eta for each sample in the batch
        f = torch.zeros(numSteps + 1, batchSize)  # Keep track of the function value for every sample at every step
        z = torch.zeros(numSteps + 1, batchSize, xShape[0], xShape[1], xShape[2])
        x = torch.zeros(numSteps + 1, batchSize, xShape[0], xShape[1], xShape[2])
        x[0] = xData  # Initalize the starting adversarial example as the clean example
        xBest = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2])  # Best adversarial example thus far
        fBest = torch.zeros(batchSize)  # Best value of the objective function thus far
        # Do the attack for a number of steps
        for k in range(0, numSteps):
            lossIndividual = torch.nn.CrossEntropyLoss(reduction='none')
            # First attack step handled slightly differently
            if k == 0:
                xKGrad = GetModelGradient(device, model, x[k], yK)  # Get the model gradient
                if targeted == True:
                    raise ValueError("Targeted Auto-Attack not yet implemented.")
                else:  # targeted is false
                    for b in range(0, batchSize):
                        x[1, b] = x[0, b] + eta[k, b] * torch.sign(
                            xKGrad[b]).cpu()  # here we use index 1 because the 0th index is the clean sample
                # Apply the projection operation and clipping to make sure xAdv does not go out of the adversarial bounds
                for b in range(0, batchSize):
                    x[1, b] = torch.clamp(ProjectionOperation(x[1, b], x[0, b], epsilonMax), min=clipMin, max=clipMax)
                # Check which adversarial x is better, the clean x or the new adversarial x
                outputsOriginal = model(x[k].to(device))
                model.zero_grad()
                f[0] = lossIndividual(outputsOriginal,
                                      yK).cpu().detach()  # Store the value in the objective function array
                outputs = model(x[k + 1].to(device))
                model.zero_grad()
                f[1] = lossIndividual(outputs, yK).cpu().detach()  # Store the value in the objective function array
                for b in range(0, batchSize):
                    # In the untargeted case we want the cost to increase
                    if f[k + 1, b] >= f[k, b] and targeted == False:
                        xBest[b] = x[k + 1, b]
                        fBest[b] = f[k + 1, b]
                    # In the targeted case we want the cost to decrease
                    elif f[k + 1, b] <= f[k, b] and targeted == True:
                        xBest[b] = x[k + 1, b]
                        fBest[b] = f[k + 1, b]
                    # Give a non-zero step size for the next iteration
                    eta[k + 1, b] = eta[k, b]
            # Not the first iteration of the attack
            else:
                xKGrad = GetModelGradient(device, model, x[k], yK)
                if targeted == True:
                    raise ValueError("Didn't implement targeted auto attack yet.")
                else:
                    for b in range(0, batchSize):
                        # Compute zk
                        z[k, b] = x[k, b] + eta[k, b] * torch.sign(xKGrad[b]).cpu()
                        z[k, b] = ProjectionOperation(z[k, b], x[0, b], epsilonMax)
                        # Compute x(k+1) using momentum
                        x[k + 1, b] = x[k, b] + alpha * (z[k, b] - x[k, b]) + (1 - alpha) * (x[k, b] - x[k - 1, b])
                        x[k + 1, b] = ProjectionOperation(x[k + 1, b], x[0, b], epsilonMax)
                        # Apply the clipping operation to make sure xAdv remains in the valid image range
                        x[k + 1, b] = torch.clamp(x[k + 1, b], min=clipMin, max=clipMax)
                # Check which x is better
                outputs = model(x[k + 1].to(device))
                model.zero_grad()
                f[k + 1] = lossIndividual(outputs, yK).cpu().detach()
                for b in range(0, batchSize):
                    # In the untargeted case we want the cost to increase
                    if f[k + 1, b] >= fBest[b] and targeted == False:
                        xBest[b] = x[k + 1, b]
                        fBest[b] = f[k + 1, b]
                # Now time to do the conditional check to possibly update the step size
                if k in wList:
                    # print(k) #For debugging
                    checkPointIndex = wList.index(k)  # Get the index of the currentCheckpoint
                    # Go through each element in the batch
                    for b in range(0, batchSize):
                        conditionOneBoolean = CheckConditionOne(f[:, b], checkPointIndex, wList, targeted)
                        conditionTwoBoolean = CheckConditionTwo(f[:, b], eta[:, b], checkPointIndex, wList)
                        # If either condition is true halve the step size, else use the step size of the last iteration
                        if conditionOneBoolean == True or conditionTwoBoolean == True:
                            eta[k + 1, b] = eta[k, b] / 2.0
                        else:
                            eta[k + 1, b] = eta[k, b]
                # If we don't need to check the conditions, just repeat the previous iteration's step size
                else:
                    for b in range(0, batchSize):
                        eta[k + 1, b] = eta[k, b]
                        # Memory clean up
            del lossIndividual
            del outputs
            torch.cuda.empty_cache()
            # Save the adversarial images from the batch
        for i in range(0, batchSize):
            # print("==========")
            # print(eta[:,i])
            # print("==========")
            xAdv[advSampleIndex] = xBest[i]
            yClean[advSampleIndex] = yData[i]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
    # All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size,
                                       randomizer=None)  # use the same batch size as the original loader
    return advLoader

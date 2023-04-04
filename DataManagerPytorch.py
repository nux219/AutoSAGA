import torchvision.datasets as datasets
import math 
import random 
import numpy
from snn_models.spiking_datasets import *
import matplotlib.pyplot as plt
import os


#Convert an image dataloader (I) to a repeat encoding dataloader (E)
def DataLoaderIToDataLoaderRE(dataLoaderI, length):
    #First convert the image dataloader to tensor form
    xTensor, yTensor = DataLoaderToTensor(dataLoaderI)
    #Create memory for the new tensor with repeat encoding
    xTensorRepeat = torch.zeros(xTensor.shape + (length,))
    #Go through and fill in the new array, probably a faster way to do this with Pytorch tensors
    for i in range(0, xTensor.shape[0]):
        for j in range(0, length):
            xTensorRepeat[i, :, :, :, j] = xTensor[i]
    #New tensor is filled in, convert back to dataloader
    dataLoaderRE = TensorToDataLoader(xTensorRepeat, yTensor, transforms=None, batchSize =dataLoaderI.batch_size, randomizer = None)
    return dataLoaderRE

#Convert an image dataloader (I) to a repeat encoding dataloader (E)
def DataLoaderREToDataLoaderI(dataLoaderRE):
    #First convert the repeated dataloader to tensor form
    xTensor, yTensor = DataLoaderToTensor(dataLoaderRE)
    #Create memory for the new tensor with repeat encoding
    xTensorImages = torch.zeros(xTensor.shape[0], xTensor.shape[1], xTensor.shape[2], xTensor.shape[3])
    #Go through and fill in the new array, probably a faster way to do this with Pytorch tensors
    for i in range(0, xTensor.shape[0]):
        xTensorImages[i] = xTensor[i, :, :, :, 0] #Just take the first image from the repeated tensor because they should be the same
    #New tensor is filled in, convert back to dataloader
    dataLoaderI = TensorToDataLoader(xTensorImages, yTensor, transforms=None, batchSize =dataLoaderRE.batch_size, randomizer = None)
    return dataLoaderI

def CheckCudaMem():
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("Unfree Memory=", a)


#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

#Validate using a dataloader 
def validateD(valLoader, model, device=None):
    #switch to evaluate mode
    model.eval()
    acc = 0 
    batchTracker = 0
    if device == None:  # assume cuda
        device = 'cuda'
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            # print(input.min(), input.max())
            #print("Processing up to sample=", batchTracker)
            # if device == None: #assume cuda
            #     inputVar = input.cuda()
            #     target = target.cuda()
            # else:
            inputVar = input.to(device)
            target = target.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
            acc += correct_k
            # for j in range(0, sampleSize):
            #     if output[j].argmax(axis=0) == target[j]:
            #         acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

#Method to validate data using Pytorch tensor inputs and a Pytorch model 
def validateT(xData, yData, model, batchSize=None):
    acc = 0 #validation accuracy 
    numSamples = xData.shape[0]
    model.eval() #change to eval mode
    if batchSize == None: #No batch size so we can feed everything into the GPU
         output = model(xData)
         for i in range(0, numSamples):
             if output[i].argmax(axis=0) == yData[i]:
                 acc = acc+ 1
    else: #There are too many samples so we must process in batch
        numBatches = int(math.ceil(xData.shape[0] / batchSize)) #get the number of batches and type cast to int
        for i in range(0, numBatches): #Go through each batch 
            print(i)
            modelOutputIndex = 0 #reset output index
            startIndex = i*batchSize
            #change the end index depending on whether we are on the last batch or not:
            if i == numBatches-1: #last batch so go to the end
                endIndex = numSamples
            else: #Not the last batch so index normally
                endIndex = (i+1)*batchSize
            output = model(xData[startIndex:endIndex])
            for j in range(startIndex, endIndex): #check how many samples in the batch match the target
                if output[modelOutputIndex].argmax(axis=0) == yData[j]:
                    acc = acc+ 1
                modelOutputIndex = modelOutputIndex + 1 #update the output index regardless
    #Do final averaging and return 
    acc = acc / numSamples
    return acc

#Input a dataloader and model
#Instead of returning a model, output is array with 1.0 dentoting the sample was correctly identified
def validateDA(valLoader, model, device=None):
    numSamples = len(valLoader.dataset)
     #variable for keep tracking of the correctly identified samples
    #switch to evaluate mode
    model.eval()

    accuracy = 0
    if device == None:
        device = 'cuda'
    # indexer = torch.zeros(1, device=device, dtype=int)
    indexer = 0
    accuracyArray = torch.zeros(numSamples).to(device)
    sampleSize = valLoader.batch_size
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            # sampleSize = input.shape[0] #Get the number of samples used in each batch
            # batchTracker = batchTracker + sampleSize
            # print("Processing up to sample=", batchTracker)
            # if device == None: #assume CUDA by default
            #     inputVar = input.cuda()
            #     target = target.cuda()
            # else:
            inputVar = input.to(device) #use the prefered device if one is specified
            target = target.to(device) #use the prefered device if one is specified
            #compute output
            output = model(inputVar)
            output = output.float()
            outmax = output.argmax(axis=1)
            correct = outmax.eq(target)
            accuracy = accuracy + correct.sum()
            # accuracyArray[indexer: indexer+sampleSize] = correct.to(torch.float)
            accuracyArray[indexer: indexer+sampleSize] = correct
            indexer = indexer + sampleSize
            #Go through and check how many samples correctly identified
            # for j in range(0, sampleSize):
            #     if output[j].argmax(axis=0) == target[j]:
            #         accuracyArray[indexer] = 1.0 #Mark with a 1.0 if sample is correctly identified
            #         accuracy = accuracy + 1
            #     indexer = indexer + 1 #update the indexer regardless
    accuracy = accuracy/numSamples
    print("Accuracy:", accuracy)
    return accuracyArray

#Replicate TF's predict method behavior 
def predictD(dataLoader, numClasses, model, device=None):
    numSamples = len(dataLoader.dataset)
    yPred = torch.zeros(numSamples, numClasses)
    #switch to evaluate mode
    # model.cuda()
    model.eval()
    indexer = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None:
                inputVar = input.cuda()
                model = model.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            # output = output[1].float()
            for j in range(0, sampleSize):
                yPred[indexer] = output[j]
                indexer = indexer + 1 #update the indexer regardless 
    return yPred

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    if sampleShape.__len__() == 4:
        xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2], sampleShape[3])
    else:
        xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData 

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape


#Show 20 images, 10 in first and row and 10 in second row
def ShowImages(xFirst, xSecond):
    n = 10  # how many digits we will display
    plt.figure(figsize=(5, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(xFirst[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(xSecond[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()



#This method randomly creates fake labels for the attack 
#The fake target is guaranteed to not be the same as the original class label 
def GenerateTargetsLabelRandomly(yData, numClasses):
    fTargetLabels=torch.zeros(len(yData))
    for i in range(0, len(yData)):
        targetLabel=random.randint(0,numClasses-1)
        while targetLabel==yData[i]:#Target and true label should not be the same 
            targetLabel=random.randint(0,numClasses-1) #Keep flipping until a different label is achieved 
        fTargetLabels[i]=targetLabel
    return fTargetLabels

#Return the first n correctly classified examples from a model 
#Note examples may not be class balanced 
def GetFirstCorrectlyIdentifiedExamples(device, dataLoader, model, numSamples):
    #First check how many samples in the dataset
    numSamplesTotal = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    xClean = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    yClean = torch.zeros(numSamples)
    #switch to evaluate mode
    model.eval()
    acc = 0 
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            batchSize = input.shape[0] #Get the number of samples used in each batch
            inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, batchSize):
                #Add the sample if it is correctly identified and we are not at the limit
                if output[j].argmax(axis=0) == target[j] and sampleIndex<numSamples: 
                    xClean[sampleIndex] = input[j]
                    yClean[sampleIndex] = target[j]
                    sampleIndex = sampleIndex+1
    #Done collecting samples, time to covert to dataloader 
    cleanLoader = TensorToDataLoader(xClean, yClean, transforms=None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanLoader


# def GetIMAGENETValidation(imgSize=32, batchSize=128, mean=(0.485, 0.456, 0.406) , std = (0.229, 0.224, 0.225)):
def GetIMAGENETValidation(imgSize=224, batchSize=128):
    traindir    = os.path.join('/mnt/mnt/ImageNet/', 'train')
    valdir      = os.path.join('/mnt/mnt/ImageNet/', 'val')
    # normalize   = transforms.Normalize(mean, std)
    # trainset    = datasets.ImageFolder(
    #                     traindir,
    #                     transforms.Compose([
    #                         transforms.RandomResizedCrop(224),
    #                         transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         normalize
    #                     ]))
    if imgSize == 224:
        testset     = datasets.ImageFolder(
                            valdir,
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(imgSize),
                                transforms.ToTensor()
                                # normalize
                            ]))
    elif imgSize == 512:
        testset     = datasets.ImageFolder(
                            valdir,
                            transforms.Compose([
                                transforms.Resize((imgSize,imgSize)),
                                transforms.ToTensor()
                                # normalize
                            ]))
    # # train_loader    = DataLoader(trainset, batch_size=batchSize, shuffle=True)
    # # n = len(testset)  # total number of examples
    # # n_test = int(0.4 * n)  # take ~10% for test
    # # test_set = torch.utils.data.Subset(testset, range(n_test))
    # r = np.arange(50000)
    # np.random.shuffle(r)
    # test_set = []
    # for i in range(12000):
    #     test_set.append(testset[r[i]])
    # valLoader     = DataLoader(test_set, batch_size=batchSize, shuffle=False)
    # del test_set
    valLoader = DataLoader(testset, batch_size=batchSize, shuffle=False)
    return valLoader


#This data is in the range 0 to 1
def GetCIFAR10Validation(imgSize = 32, batchSize=128, mean=0.5):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[mean, mean, mean], std=[0.5,0.5,0.5]),
        transforms.Normalize(mean=[mean, mean, mean], std=[1.0, 1.0, 1.0]),

    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

def GetCIFAR10Validation_norm(imgSize = 32, batchSize=128, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[mean, mean, mean], std=[0.5,0.5,0.5]),
        transforms.Normalize(mean=mean, std=std),

    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=False)
    return valLoader


def GetCIFAR100Validation_norm(imgSize = 32, batchSize=128, mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[mean, mean, mean], std=[0.5,0.5,0.5]),
        transforms.Normalize(mean=mean, std=std),

    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=False)
    return valLoader

def GetCIFAR100Validation(imgSize = 32, batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[mean, mean, mean], std=[0.5,0.5,0.5]),
        # transforms.Normalize(mean=mean, std=std),

    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=False)
    return valLoader

def GetCIFAR10Validation_unnormalize(imgSize = 32, batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=False)
    return valLoader


def GetCIFAR10Validation_snn(imgSize = 32, batchSize=128, length=20):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[mean, mean, mean], std=[1.0, 1.0, 1.0]),
    ])
    val_data = Image_Dataset_Adapter(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), length=length, mode='repeat', max_rate=1)
    valLoader = torch.utils.data.DataLoader(val_data, batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

def GetCIFAR100Validation_snn(imgSize = 32, batchSize=128, length=20):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[mean, mean, mean], std=[1.0, 1.0, 1.0]),
    ])
    val_data = Image_Dataset_Adapter(datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform=transformTest), length=length, mode='repeat', max_rate=1)
    valLoader = torch.utils.data.DataLoader(val_data, batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader


def GetCIFAR100Validation_resize(imgSizeH=160, imgSizeW=128, batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((imgSizeH, imgSizeW)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[mean, mean, mean], std=[0.5,0.5,0.5]),
        # transforms.Normalize(mean=mean, std=std),

    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=False)
    return valLoader

def GetCIFAR10Validation_unnormalize_resize(imgSizeH=160, imgSizeW=128, batchSize=128, mean=0.5):
    transformTest = transforms.Compose([
        transforms.Resize((imgSizeH, imgSizeW)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[mean, mean, mean], std=[1.0, 1.0, 1.0]),
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

def GetCIFAR10Training(imgSize = 32, batchSize=128, mean=0.5):
    toTensorTransform = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[mean, mean, mean], std=[0.5,0.5,0.5]),
        # transforms.Normalize(mean=[mean, mean, mean], std=[1.0, 1.0, 1.0]),

    ])
    trainLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=toTensorTransform), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return trainLoader

def GetCIFAR10Training_snn(imgSize = 32, batchSize=128, mean=0.5, length=20):
    toTensorTransform = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[mean, mean, mean], std=[0.5,0.5,0.5]),
        # transforms.Normalize(mean=[mean, mean, mean], std=[1.0, 1.0, 1.0]),

    ])
    train_data = Image_Dataset_Adapter(datasets.CIFAR10(root='./data', train=True, download=True, transform=toTensorTransform), length=length, mode='repeat', max_rate=1)
    trainLoaderSNN = torch.utils.data.DataLoader(train_data, batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    trainLoader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True, transform=toTensorTransform), batch_size=batchSize,
        shuffle=False, num_workers=1, pin_memory=True)
    return trainLoaderSNN, trainLoader

def GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, dataLoader, numClasses):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = predictD(dataLoader, numClasses, model)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

def GetCorrectlyIdentifiedSamplesBalancedDefense(defense, totalSamplesRequired, dataLoader, numClasses):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses)
    # print('sampleShape: ', sampleShape)
    # print('xData.shape: ',xData.shape)
    # print('sampleShape.__len__ : ',sampleShape.__len__() )
    print('numSamplesPerClass: ', numSamplesPerClass)
    if sampleShape.__len__() == 4:
        correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2], sampleShape[3]))
        # print(correctlyClassifiedSamples.shape)
    else:
        correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = defense.predictD(dataLoader, numClasses)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            # raise ValueError("The network does not have enough correctly predicted samples for this class.")
            print("The network does not have enough correctly predicted samples for this class: ", c)
    #Assume we have enough samples now, restore in a properly shaped array
    xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    if sampleShape.__len__() == 4:
        xCorrect_snn = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3], xData.shape[4]))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it
            if sampleShape.__len__() == 4:
                xCorrect[currentIndex] = correctlyClassifiedSamples[c,j,:,:,:,-1]
                xCorrect_snn[currentIndex] = correctlyClassifiedSamples[c,j]
            else:
                xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms=None, batchSize=dataLoader.batch_size,
                                                 randomizer=None)
    # print(xCorrect[0,0,:])
    # print(xCorrect.shape)
    # print(xCorrect_snn[0,0,:,:,-1])
    # print(xCorrect_snn[0,0,:,:,0])
    # if sampleShape.__len__() == 4:
    #     cleanDataLoader_snn = TensorToDataLoader(xCorrect_snn, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    #     return cleanDataLoader, cleanDataLoader_snn
    return cleanDataLoader

def GetCorrectlyIdentifiedSamplesBalancedDefense_snn(defense, totalSamplesRequired, dataLoader, attackloader, numClasses):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    xData_att, yData_att = DataLoaderToTensor(attackloader)
    #Basic error checking
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses)
    correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = defense.predictD(dataLoader, numClasses)
    for i in range(0, xData.shape[0]): #Go through every sample
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0)
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            # correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData_att[i] #Save the sample
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array
    xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader


# Do the computation from scratch to get the correctly identified overlapping examples
# Note these samples will be the same size as the input size required by model A
def GetFirstCorrectlyOverlappingSamplesBalanced(device, sampleNum, numClasses, dataLoader, modelPlusA, modelPlusB):
    # First check if modelA needs resize
    xTestOrig, yTestOrig = DataLoaderToTensor(dataLoader)
    # We need to resize first
    if modelPlusB.imgSizeH != xTestOrig.shape[2] or modelPlusB.imgSizeW != xTestOrig.shape[3]:
        xTestOrigResize = torch.zeros(xTestOrig.shape[0], 3, modelPlusB.imgSizeH, modelPlusB.imgSizeW)
        rs = transforms.Resize((modelPlusB.imgSizeH, modelPlusB.imgSizeW))  # resize the samples for model A
        # Go through every sample
        for i in range(0, xTestOrig.shape[0]):
            xTestOrigResize[i] = rs(xTestOrig[i])  # resize to match dimensions required by modelA
        # Make a new dataloader
        dataLoaderB = TensorToDataLoader(xTestOrigResize, yTestOrig, transforms=None, batchSize=dataLoader.batch_size,
                                        randomizer=None)
        accArrayB = modelPlusB.validateDA(dataLoaderB)
    else:
        dataLoaderB = dataLoader
        accArrayB = modelPlusB.validateDA(dataLoader)
    # Get accuracy array for each model
    accArrayA = modelPlusA.validateDA(dataLoader)
    # accArrayB = modelPlusB.validateDA(dataLoaderB)
    accArray = accArrayA + accArrayB  # will be 2 if both classifers recognize the sample correctly
    # Get the total number of samples
    totalSampleNum = accArrayA.shape[0]
    # Do some basic error checking
    if sampleNum % numClasses != 0:
        raise ValueError("Number of samples not divisable by the number of classes")
    # Basic variable setup
    samplePerClassCount = torch.zeros(numClasses)  # keep track of samples per class
    maxRequireSamplesPerClass = int(sampleNum / numClasses)  # Max number of samples we need per class
    xTest, yTest = DataLoaderToTensor(dataLoader)  # Get all the data as tensors
    xTest_B, yTest_B = DataLoaderToTensor(dataLoaderB)  # Get all the data as tensors
    # Memory for the solution
    xClean = torch.zeros(sampleNum, 3, modelPlusA.imgSizeH, modelPlusA.imgSizeW)
    xClean_B = torch.zeros(sampleNum, 3, modelPlusB.imgSizeH, modelPlusB.imgSizeW)
    yClean = torch.zeros(sampleNum)
    sampleIndexer = 0
    # rs = torchvision.transforms.Resize((modelPlusA.imgSizeH, modelPlusA.imgSizeW)) #resize the samples for model A
    # Go through all the samples
    for i in range(0, totalSampleNum):
        currentClass = int(yTest[i])
        # Check to make sure all classifiers identify the sample correctly AND we don't have enough of this class yet
        if accArray[i] == 2.0 and samplePerClassCount[currentClass] < maxRequireSamplesPerClass:
            # xClean[sampleIndexer] = rs(xTest[i]) #resize to match dimensions required by modelA
            xClean[sampleIndexer] = xTest[i]
            xClean_B[sampleIndexer] = xTest_B[i]
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
            print('samplePerClassCount[i]: ', samplePerClassCount[i])
            raise ValueError("We didn't find enough of class: " + str(i))
    # Conver the solution into a dataloader
    # Find the smallest batch size to avoid running out of memory
    minBatchSize = int(numpy.minimum(modelPlusA.batchSize, modelPlusB.batchSize))
    cleanDataLoader_B = TensorToDataLoader(xClean_B, yClean, transforms=None, batchSize=minBatchSize, randomizer=None)
    cleanDataLoader = TensorToDataLoader(xClean, yClean, transforms=None, batchSize=minBatchSize, randomizer=None)
    # Do one last check to make sure all samples identify the clean loader correctly
    cleanAccA = modelPlusA.validateD(cleanDataLoader)
    cleanAccB = modelPlusB.validateD(cleanDataLoader)
    cleanAcc_B = modelPlusB.validateD(cleanDataLoader_B)
    if cleanAccA != 1.0 or cleanAccB != 1.0 or cleanAcc_B != 1.0:
        print("Clean Acc " + modelPlusA.modelName + ":", cleanAccA)
        print("Clean Acc " + modelPlusB.modelName + ":", cleanAccB)
        print("Clean Acc " + modelPlusB.modelName + ":", cleanAcc_B)
        # raise ValueError("The clean accuracy is not 1.0 for both models.")
        print("The clean accuracy is not 1.0 for both models.")
    # All error checking done, return the clean balanced loader
    if modelPlusB.imgSizeH != xTestOrig.shape[2] or modelPlusB.imgSizeW != xTestOrig.shape[3]:
        return cleanDataLoader, cleanDataLoader_B
    else:
        return cleanDataLoader


# Do the computation from scratch to get the correctly identified overlapping examples
# Note these samples will be the same size as the input size required by model A
def GetFirstCorrectlyOverlappingSamplesBalanced_SNN(device, sampleNum, numClasses, dataLoaderA, dataLoaderB, modelPlusA,
                                                    modelPlusB, mean, std):
    # First check if modelA needs resize
    xTestOrig, yTestOrig = DataLoaderToTensor(dataLoaderA)
    # We need to resize first
    if modelPlusA.imgSizeH != xTestOrig.shape[2] or modelPlusA.imgSizeW != xTestOrig.shape[3]:
        xTestOrigResize = torch.zeros(xTestOrig.shape[0], 3, modelPlusA.imgSizeH, modelPlusA.imgSizeW)
        rs = transforms.Resize((modelPlusA.imgSizeH, modelPlusA.imgSizeW))  # resize the samples for model A
        # Go through every sample
        for i in range(0, xTestOrig.shape[0]):
            xTestOrigResize[i] = rs(xTestOrig[i])  # resize to match dimensions required by modelA
        # Make a new dataloader
        dataLoader = TensorToDataLoader(xTestOrigResize, yTestOrig, transforms=None, batchSize=dataLoaderA.batch_size,
                                        randomizer=None)
    # Get accuracy array for each model
    accArrayA = modelPlusA.validateDA(dataLoaderA)
    accArrayB = modelPlusB.validateDA(dataLoaderB)
    accArray = accArrayA + accArrayB  # will be 2 if both classifers recognize the sample correctly
    # Get the total number of samples
    totalSampleNum = accArrayA.shape[0]
    # Do some basic error checking
    if sampleNum % numClasses != 0:
        raise ValueError("Number of samples not divisable by the number of classes")
    # Basic variable setup
    samplePerClassCount = torch.zeros(numClasses)  # keep track of samples per class
    maxRequireSamplesPerClass = int(sampleNum / numClasses)  # Max number of samples we need per class
    xTest, yTest = DataLoaderToTensor(dataLoaderB)  # Get all the data as tensors
    # Memory for the solution
    xClean = torch.zeros(sampleNum, 3, modelPlusA.imgSizeH, modelPlusA.imgSizeW)
    yClean = torch.zeros(sampleNum)
    sampleIndexer = 0
    # rs = torchvision.transforms.Resize((modelPlusA.imgSizeH, modelPlusA.imgSizeW)) #resize the samples for model A
    # Go through all the samples
    for i in range(0, totalSampleNum):
        currentClass = int(yTest[i])
        # Check to make sure all classifiers identify the sample correctly AND we don't have enough of this class yet
        if accArray[i] == 2.0 and samplePerClassCount[currentClass] < maxRequireSamplesPerClass:
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
            raise ValueError("We didn't find enough of class: " + str(i))
    # Conver the solution into a dataloader
    # Find the smallest batch size to avoid running out of memory
    minBatchSize = int(numpy.minimum(modelPlusA.batchSize, modelPlusB.batchSize))
    cleanDataLoaderA = TensorToDataLoader(xClean, yClean,
                                         transforms=transforms.Normalize(mean=mean, std=std),
                                         batchSize=minBatchSize, randomizer=None)
    cleanDataLoader = TensorToDataLoader(xClean, yClean, transforms=None, batchSize=minBatchSize, randomizer=None)
    # Do one last check to make sure all samples identify the clean loader correctly
    cleanAccA = modelPlusA.validateD(cleanDataLoaderA)
    cleanAccB = modelPlusB.validateD(cleanDataLoader)
    if cleanAccA != 1.0 or cleanAccB != 1.0:
        print("Clean Acc " + modelPlusA.modelName + ":", cleanAccA)
        print("Clean Acc " + modelPlusB.modelName + ":", cleanAccB)
        print("The clean accuracy is not 1.0 for both models.")
        # raise ValueError("The clean accuracy is not 1.0 for both models.")

    # All error checking done, return the clean balanced loader
    return cleanDataLoaderA, cleanDataLoader

def GetFirstCorrectlyOverlappingSamplesBalanced_SNN_v2(device, sampleNum, numClasses, dataLoaderA, dataLoaderB, modelPlusA,
                                                    modelPlusB):
    # First check if modelA needs resize
    xTestOrig, yTestOrig = DataLoaderToTensor(dataLoaderA)
    # We need to resize first
    if modelPlusA.imgSizeH != xTestOrig.shape[2] or modelPlusA.imgSizeW != xTestOrig.shape[3]:
        xTestOrigResize = torch.zeros(xTestOrig.shape[0], 3, modelPlusA.imgSizeH, modelPlusA.imgSizeW)
        rs = transforms.Resize((modelPlusA.imgSizeH, modelPlusA.imgSizeW))  # resize the samples for model A
        # Go through every sample
        for i in range(0, xTestOrig.shape[0]):
            xTestOrigResize[i] = rs(xTestOrig[i])  # resize to match dimensions required by modelA
        # Make a new dataloader
        dataLoader = TensorToDataLoader(xTestOrigResize, yTestOrig, transforms=None, batchSize=dataLoaderA.batch_size,
                                        randomizer=None)
    # Get accuracy array for each model
    accArrayA = modelPlusA.validateDA(dataLoaderA)
    accArrayB = modelPlusB.validateDA(dataLoaderB)
    accArray = accArrayA + accArrayB  # will be 2 if both classifers recognize the sample correctly
    # Get the total number of samples
    totalSampleNum = accArrayA.shape[0]
    # Do some basic error checking
    if sampleNum % numClasses != 0:
        raise ValueError("Number of samples not divisable by the number of classes")
    # Basic variable setup
    samplePerClassCount = torch.zeros(numClasses)  # keep track of samples per class
    maxRequireSamplesPerClass = int(sampleNum / numClasses)  # Max number of samples we need per class
    xTestA, yTestA = DataLoaderToTensor(dataLoaderA)  # Get all the data as tensors
    xTest, yTest = DataLoaderToTensor(dataLoaderB)  # Get all the data as tensors
    # Memory for the solution
    xClean = torch.zeros(sampleNum, 3, modelPlusA.imgSizeH, modelPlusA.imgSizeW)
    yClean = torch.zeros(sampleNum)
    xCleanA = torch.zeros(sampleNum, 3, modelPlusA.imgSizeH, modelPlusA.imgSizeW)
    yCleanA = torch.zeros(sampleNum)
    sampleIndexer = 0
    # rs = torchvision.transforms.Resize((modelPlusA.imgSizeH, modelPlusA.imgSizeW)) #resize the samples for model A
    # Go through all the samples
    for i in range(0, totalSampleNum):
        currentClass = int(yTest[i])
        # Check to make sure all classifiers identify the sample correctly AND we don't have enough of this class yet
        if accArray[i] == 2.0 and samplePerClassCount[currentClass] < maxRequireSamplesPerClass:
            # xClean[sampleIndexer] = rs(xTest[i]) #resize to match dimensions required by modelA
            xClean[sampleIndexer] = xTest[i]
            yClean[sampleIndexer] = yTest[i]
            xCleanA[sampleIndexer] = xTestA[i]
            yCleanA[sampleIndexer] = yTestA[i]
            sampleIndexer = sampleIndexer + 1  # update the indexer
            samplePerClassCount[currentClass] = samplePerClassCount[
                                                    currentClass] + 1  # Update the number of samples for this class
    # Check the over all number of samples as well
    if sampleIndexer != sampleNum:
        print("Not enough clean samples found.")
    # Do some error checking on the classes
    for i in range(0, numClasses):
        if samplePerClassCount[i] != maxRequireSamplesPerClass:
            raise ValueError("We didn't find enough of class: " + str(i))
    # Conver the solution into a dataloader
    # Find the smallest batch size to avoid running out of memory
    minBatchSize = int(numpy.minimum(modelPlusA.batchSize, modelPlusB.batchSize))
    cleanDataLoaderA = TensorToDataLoader(xCleanA, yCleanA,
                                         batchSize=minBatchSize, randomizer=None)
    cleanDataLoader = TensorToDataLoader(xClean, yClean, transforms=None, batchSize=minBatchSize, randomizer=None)
    # Do one last check to make sure all samples identify the clean loader correctly
    cleanAccA = modelPlusA.validateD(cleanDataLoaderA)
    cleanAccB = modelPlusB.validateD(cleanDataLoader)
    if cleanAccA != 1.0 or cleanAccB != 1.0:
        print("Clean Acc " + modelPlusA.modelName + ":", cleanAccA)
        print("Clean Acc " + modelPlusB.modelName + ":", cleanAccB)
        print("The clean accuracy is not 1.0 for both models.")
        # raise ValueError("The clean accuracy is not 1.0 for both models.")

    # All error checking done, return the clean balanced loader
    return cleanDataLoader


# dataLoaderA: SNN's dataset
# dataLoaderB: attack model's dataset
# modelPlusA: SNN
# modelPlusB: attack model
def GetFirstCorrectlyOverlappingSamplesBalanced_SNN_diffimgSize(device, sampleNum, numClasses, dataLoaderA, dataLoaderB, modelPlusA,
                                                    modelPlusB):
    # First check if modelA needs resize
    xTestOrig, yTestOrig = DataLoaderToTensor(dataLoaderB)
    # We need to resize first
    # if modelPlusA.imgSizeH != xTestOrig.shape[2] or modelPlusA.imgSizeW != xTestOrig.shape[3]:
    #     xTestOrigResize = torch.zeros(xTestOrig.shape[0], 3, modelPlusA.imgSizeH, modelPlusA.imgSizeW)
    #     rs = transforms.Resize((modelPlusA.imgSizeH, modelPlusA.imgSizeW))  # resize the samples for model A
    #     # Go through every sample
    #     for i in range(0, xTestOrig.shape[0]):
    #         xTestOrigResize[i] = rs(xTestOrig[i])  # resize to match dimensions required by modelA
    #     # Make a new dataloader
    #     dataLoaderA = TensorToDataLoader(xTestOrigResize, yTestOrig, transforms=None, batchSize=dataLoaderA.batch_size,
    #                                     randomizer=None)
    # Get accuracy array for each model
    accArrayA = modelPlusA.validateDA(dataLoaderA)
    accArrayB = modelPlusB.validateDA(dataLoaderB)
    accArray = accArrayA + accArrayB  # will be 2 if both classifers recognize the sample correctly
    # Get the total number of samples
    totalSampleNum = accArrayA.shape[0]
    # Do some basic error checking
    if sampleNum % numClasses != 0:
        raise ValueError("Number of samples not divisable by the number of classes")
    # Basic variable setup
    samplePerClassCount = torch.zeros(numClasses)  # keep track of samples per class
    maxRequireSamplesPerClass = int(sampleNum / numClasses)  # Max number of samples we need per class
    xTestA, yTestA = DataLoaderToTensor(dataLoaderA)  # Get all the data as tensors
    xTest, yTest = DataLoaderToTensor(dataLoaderB)  # Get all the data as tensors
    # Memory for the solution
    xClean = torch.zeros(sampleNum, 3, modelPlusB.imgSizeH, modelPlusB.imgSizeW)
    yClean = torch.zeros(sampleNum)
    xCleanA = torch.zeros(sampleNum, 3, modelPlusA.imgSizeH, modelPlusA.imgSizeW)
    yCleanA = torch.zeros(sampleNum)
    sampleIndexer = 0
    # rs = torchvision.transforms.Resize((modelPlusA.imgSizeH, modelPlusA.imgSizeW)) #resize the samples for model A
    # Go through all the samples
    for i in range(0, totalSampleNum):
        currentClass = int(yTest[i])
        # Check to make sure all classifiers identify the sample correctly AND we don't have enough of this class yet
        if accArray[i] == 2.0 and samplePerClassCount[currentClass] < maxRequireSamplesPerClass:
            # xClean[sampleIndexer] = rs(xTest[i]) #resize to match dimensions required by modelA
            xClean[sampleIndexer] = xTest[i]
            yClean[sampleIndexer] = yTest[i]
            xCleanA[sampleIndexer] = xTestA[i]
            yCleanA[sampleIndexer] = yTestA[i]
            sampleIndexer = sampleIndexer + 1  # update the indexer
            samplePerClassCount[currentClass] = samplePerClassCount[
                                                    currentClass] + 1  # Update the number of samples for this class
    # Check the over all number of samples as well
    if sampleIndexer != sampleNum:
        print("Not enough clean samples found.")
    # Do some error checking on the classes
    for i in range(0, numClasses):
        if samplePerClassCount[i] != maxRequireSamplesPerClass:
            raise ValueError("We didn't find enough of class: " + str(i))
    # Conver the solution into a dataloader
    # Find the smallest batch size to avoid running out of memory
    minBatchSize = int(numpy.minimum(modelPlusA.batchSize, modelPlusB.batchSize))
    cleanDataLoaderA = TensorToDataLoader(xCleanA, yCleanA,
                                         batchSize=minBatchSize, randomizer=None)
    cleanDataLoader = TensorToDataLoader(xClean, yClean, transforms=None, batchSize=minBatchSize, randomizer=None)
    # Do one last check to make sure all samples identify the clean loader correctly
    cleanAccA = modelPlusA.validateD(cleanDataLoaderA)
    cleanAccB = modelPlusB.validateD(cleanDataLoader)
    if cleanAccA != 1.0 or cleanAccB != 1.0:
        print("Clean Acc " + modelPlusA.modelName + ":", cleanAccA)
        print("Clean Acc " + modelPlusB.modelName + ":", cleanAccB)
        print("The clean accuracy is not 1.0 for both models.")
        # raise ValueError("The clean accuracy is not 1.0 for both models.")

    # All error checking done, return the clean balanced loader
    return cleanDataLoader

# This is the model "plus" class
# It wraps a Pytorch model, string name, and transforms together
import torch
import torchvision
import DataManagerPytorch as DMP
from spikingjelly.clock_driven import functional


class ModelPlus():
    # Constuctor arguements are self explanatory
    def __init__(self, modelName, model, device, imgSizeH, imgSizeW, batchSize):
        self.modelName = modelName
        self.model = model
        self.imgSizeH = imgSizeH
        self.imgSizeW = imgSizeW
        self.batchSize = batchSize
        self.resizeTransform = torchvision.transforms.Resize((imgSizeH, imgSizeW))
        self.device = device

    # Validate a dataset, makes sure that the dataset is the right size before processing
    def validateD(self, dataLoader):
        # Put the images in the right size if they are not already
        dataLoaderFinal = self.formatDataLoader(dataLoader)
        # Make a copy of the model and put it on the GPU
        currentModel = self.model
        currentModel.to(self.device)
        # Get the accuracy
        acc = DMP.validateD(dataLoaderFinal, currentModel)
        # Clean up the GPU memory
        del currentModel
        torch.cuda.empty_cache()
        return acc

    # Predict on a dataset, makes sure that the dataset is the right size before processing
    def predictD(self, dataLoader, numClasses):
        # Put the images in the right size if they are not already
        dataLoaderFinal = self.formatDataLoader(dataLoader)
        # Make a copy of the model and put it on the GPU
        currentModel = self.model
        currentModel.to(self.device)
        # Get the accuracy
        yPred = DMP.predictD(dataLoaderFinal, numClasses, currentModel)
        # Clean up the GPU memory
        del currentModel
        torch.cuda.empty_cache()
        return yPred

    # This has possiblity to run out of memory
    def predictT(self, xData):
        # Check to make sure it is the right shape
        if xData.shape[1] != self.imgSizeH or xData.shape[2] != self.imgSizeW:
            xFinal = self.resizeTransform(xData)
        else:
            xFinal = xData
        # Put model and data on the device
        xFinal = xFinal.to(self.device)
        currentModel = self.model
        currentModel.to(self.device)
        # Do the prediction
        yPred = currentModel(xFinal)
        # Memory clean up
        del currentModel
        del xFinal
        torch.cuda.empty_cache()
        # Return
        return yPred

    # Validate AND generate a model array
    def validateDA(self, dataLoader):
        # Put the images in the right size if they are not already
        dataLoaderFinal = self.formatDataLoader(dataLoader)
        # Make a copy of the model and put it on the GPU
        currentModel = self.model
        currentModel.to(self.device)
        # Get the accuracy
        accArray = DMP.validateDA(dataLoaderFinal, currentModel, self.device)
        # print('Accuracy: ', accArray)
        # Clean up the GPU memory
        del currentModel
        torch.cuda.empty_cache()
        return accArray

    # Makes sure the inputs are the right size
    def formatDataLoader(self, dataLoader):
        sampleShape = DMP.GetOutputShape(dataLoader)
        # Check if we need to do resizing, if not just return the original loader
        if sampleShape[1] == self.imgSizeH and sampleShape[2] == self.imgSizeW:
            return dataLoader
        else:  # We need to do resizing
            print("Resize required. Processing now.")
            p = torchvision.transforms.ToPILImage()
            t = torchvision.transforms.ToTensor()
            numSamples = len(dataLoader.dataset)
            sampleShape = DMP.GetOutputShape(dataLoader)  # Get the output shape from the dataloader
            sampleIndex = 0
            batchTracker = 0
            xData = torch.zeros(numSamples, sampleShape[0], self.imgSizeH, self.imgSizeW)
            yData = torch.zeros(numSamples)
            # Go through and process the data in batches...kind of
            for i, (input, target) in enumerate(dataLoader):
                batchSize = input.shape[0]  # Get the number of samples used in each batch
                # print("Resize processing up to=", batchTracker)
                batchTracker = batchTracker + batchSize
                # Save the samples from the batch in a separate tensor
                for batchIndex in range(0, batchSize):
                    # Convert to pil image, resize, convert back to tensor
                    # xData[sampleIndex] = t(self.resizeTransform(p(input[batchIndex])))
                    xData[sampleIndex] = self.resizeTransform(input[batchIndex])
                    yData[sampleIndex] = target[batchIndex]
                    sampleIndex = sampleIndex + 1  # increment the sample index
            # All the data has been resized, time to put in the dataloader
            newDataLoader = DMP.TensorToDataLoader(xData, yData, transforms=None, batchSize=self.batchSize,
                                                   randomizer=None)
            # Note we don't use the original batch size because the image may have become larger
            # i.e. to large to fit in GPU memory so we use the batch specified in the ModelPlus constructor
            return newDataLoader

    # Go through and delete the main parts that might take up GPU memory
    def clearModel(self):
        print("Warning, model " + self.modelName + " is being deleted and should not be called again!")
        del self.model
        torch.cuda.empty_cache()

    # Special model plus class to deal specifically with the backprop SNN


# The backprop SNN takes in a special sized tensor e.g. (3,32,32,20) but the other models use (3, 32, 32)
# Therefore this class is needed to handle the internal conversion between tensor
# It is only needed for the backprop SNN (the transfer SNNs take in (3, 32, 32))
class ModelPlusSNNRepeat():
    # Constuctor arguements are self explanatory
    def __init__(self, modelName, model, device, imgSizeH, imgSizeW, batchSize, snnLength):
        self.modelName = modelName
        self.model = model
        self.imgSizeH = imgSizeH
        self.imgSizeW = imgSizeW
        self.batchSize = batchSize
        self.resizeTransform = torchvision.transforms.Resize((imgSizeH, imgSizeW))
        self.snnLength = snnLength
        self.device = device

    def validateD(self, dataLoader):
        # Put the images in the right size if they are not already
        dataLoaderFinal = self.formatDataLoader(dataLoader)
        # Make a copy of the model and put it on the GPU
        currentModel = self.model
        currentModel.to(self.device)
        # Get the accuracy
        acc = DMP.validateD(dataLoaderFinal, currentModel)
        # Clean up the GPU memory
        del currentModel
        torch.cuda.empty_cache()
        return acc

    # Validate AND generate a model array
    def validateDA(self, dataLoader):
        # Put the images in the right size if they are not already
        dataLoaderFinal = self.formatDataLoader(dataLoader)
        # Make a copy of the model and put it on the GPU
        currentModel = self.model
        currentModel.to(self.device)
        # Get the accuracy
        accArray = DMP.validateDA(dataLoaderFinal, currentModel)
        # Clean up the GPU memory
        del currentModel
        torch.cuda.empty_cache()
        return accArray

    # Makes sure the inputs are the right size
    def formatDataLoader(self, dataLoader):
        sampleShape = DMP.GetOutputShape(dataLoader)
        # Check if we need to do resizing, if not just return the original loader
        if sampleShape[1] == self.imgSizeH and sampleShape[2] == self.imgSizeW and len(sampleShape)==3:
            return DMP.DataLoaderIToDataLoaderRE(dataLoader, self.snnLength)
        elif sampleShape[1] == self.imgSizeH and sampleShape[2] == self.imgSizeW and sampleShape[3]==self.snnLength:
            return dataLoader
        else:  # We need to do resizing
            print("Resize required. Processing now.")
            p = torchvision.transforms.ToPILImage()
            t = torchvision.transforms.ToTensor()
            numSamples = len(dataLoader.dataset)
            sampleShape = DMP.GetOutputShape(dataLoader)  # Get the output shape from the dataloader
            sampleIndex = 0
            batchTracker = 0
            xData = torch.zeros(numSamples, sampleShape[0], self.imgSizeH, self.imgSizeW, self.snnLength)
            yData = torch.zeros(numSamples)
            # Go through and process the data in batches...kind of
            for i, (input, target) in enumerate(dataLoader):
                batchSize = input.shape[0]  # Get the number of samples used in each batch
                # print("Resize processing up to=", batchTracker)
                batchTracker = batchTracker + batchSize
                # Save the samples from the batch in a separate tensor
                for batchIndex in range(0, batchSize):
                    # Convert to pil image, resize, convert back to tensor
                    xData[sampleIndex] = self.resizeTransform(input[batchIndex]).repeat(
                        (self.snnLength, 1, 1, 1)).permute((1, 2, 3, 0))
                    yData[sampleIndex] = target[batchIndex]
                    sampleIndex = sampleIndex + 1  # increment the sample index
            # All the data has been resized, time to put in the dataloader
            newDataLoader = DMP.TensorToDataLoader(xData, yData, transforms=None, batchSize=self.batchSize,
                                                   randomizer=None)
            # Note we don't use the original batch size because the image may have become larger
            # i.e. to large to fit in GPU memory so we use the batch specified in the ModelPlus constructor
            return newDataLoader

    # Go through and delete the main parts that might take up GPU memory
    def clearModel(self):
        print("Warning, model " + self.modelName + " is being deleted and should not be called again!")
        del self.model
        torch.cuda.empty_cache()

    # This is the class to use with the Spiking Jelly SNN


class ModelPlusSpikingJelly():
    # Constuctor arguements are self explanatory
    def __init__(self, modelName, model, device, imgSizeH, imgSizeW, batchSize):
        self.modelName = modelName
        self.model = model
        self.imgSizeH = imgSizeH
        self.imgSizeW = imgSizeW
        self.batchSize = batchSize
        self.resizeTransform = torchvision.transforms.Resize((imgSizeH, imgSizeW))
        self.device = device

    # # Predict on a dataset, makes sure that the dataset is the right size before processing
    # def predictD(self, dataLoader, numClasses):
    #     # Put the images in the right size if they are not already
    #     dataLoaderFinal = self.formatDataLoader(dataLoader)
    #     # Make a copy of the model and put it on the GPU
    #     currentModel = self.model
    #     currentModel.to(self.device)
    #     # Get the accuracy
    #     yPred = DMP.predictD(dataLoaderFinal, numClasses, currentModel)
    #     # Clean up the GPU memory
    #     del currentModel
    #     torch.cuda.empty_cache()
    #     return yPred

    # def validateD(self, dataLoader, device):
    #    #Put the images in the right size if they are not already
    #    dataLoaderFinal = self.formatDataLoader(dataLoader)
    #    #Make a copy of the model and put it on the GPU
    #    currentModel = self.model
    #    currentModel.to(device)
    #    #Get the accuracy
    #    acc = DMP.validateD(dataLoaderFinal, currentModel)
    #    #Clean up the GPU memory
    #    del currentModel
    #    torch.cuda.empty_cache()
    #    return acc

    # Validate using a dataloader, special for Spiking Jelly models
    def validateD(self, dataLoader):
        # Format the dataloader
        dataLoaderFinal = self.formatDataLoader(dataLoader)
        currentModel = self.model
        currentModel.to(self.device)
        # switch to evaluate mode
        currentModel.eval()
        acc = 0
        # batchTracker = 0
        if self.device == None:  # assume cuda
            self.device = 'cuda'
        with torch.no_grad():
            # Go through and process the data in batches
            for i, (input, target) in enumerate(dataLoaderFinal):
                functional.reset_net(currentModel)  # Line to reset model memory to accodomate Spiking Jelly
                # sampleSize = input.shape[0]  # Get the number of samples used in each batch
                # batchTracker = batchTracker + sampleSize
                # print("Processing up to sample=", batchTracker)
                # if self.device == None:  # assume cuda
                #     inputVar = input.cuda()
                #     target = target.cuda()
                # else:
                inputVar = input.to(self.device)
                target = target.to(self.device)
                # compute output
                output = currentModel(inputVar)
                output = output.float().mean(0)  # Special for Spiking Jelly because have to deal with spikes instead of softmax
                # Go through and check how many samples correctly identified
                _, pred = output.topk(1, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
                acc += correct_k
                # for j in range(0, sampleSize):
                #     if output[j].argmax(axis=0) == target[j]:
                #         acc = acc + 1
        acc = acc / float(len(dataLoaderFinal.dataset))
        # Clean up the GPU memory
        del currentModel
        torch.cuda.empty_cache()
        return acc

    ##Validate AND generate a model array
    # def validateDA(self, dataLoader, device):
    #    #Put the images in the right size if they are not already
    #    dataLoaderFinal = self.formatDataLoader(dataLoader)
    #    #Make a copy of the model and put it on the GPU
    #    currentModel = self.model
    #    currentModel.to(device)
    #    #Get the accuracy
    #    accArray = self.validateDA(dataLoaderFinal, currentModel)
    #    #Clean up the GPU memory
    #    del currentModel
    #    torch.cuda.empty_cache()
    #    return accArray

    # Input a dataloader and model
    # Instead of returning a model, output is array with 1.0 dentoting the sample was correctly identified
    def validateDA(self, dataLoader):
        # Put the images in the right size if they are not already
        dataLoaderFinal = self.formatDataLoader(dataLoader)
        # Make a copy of the model and put it on the GPU
        currentModel = self.model
        currentModel.to(self.device)
        numSamples = len(dataLoaderFinal.dataset)

        # switch to evaluate mode
        currentModel.eval()
        indexer = 0
        accuracy = 0
        batchTracker = 0
        if self.device == None:  # assume cuda
            self.device = 'cuda'
        accuracyArray = torch.zeros(numSamples).to(self.device)  # variable for keep tracking of the correctly identified samples
        sampleSize = dataLoaderFinal.batch_size
        with torch.no_grad():
            # Go through and process the data in batches
            for i, (input, target) in enumerate(dataLoaderFinal):
                functional.reset_net(currentModel)  # Line to reset model memory to accodomate Spiking Jelly
                 # Get the number of samples used in each batch
                batchTracker = batchTracker + sampleSize
                print("Processing up to sample=", batchTracker)
                # if self.device == None:  # assume CUDA by default
                #     inputVar = input.cuda()
                #     target = target.cuda()
                # else:
                inputVar = input.to(self.device)  # use the prefered device if one is specified
                target = target.to(self.device)  # use the prefered device if one is specified
                # compute output
                output = currentModel(inputVar)
                output = output.float().mean(0)  # Spiking Jelly change
                # Go through and check how many samples correctly identified
                outmax = output.argmax(axis=1)
                correct = outmax.eq(target)
                accuracy = accuracy + correct.sum()
                accuracyArray[indexer: indexer + sampleSize] = correct
                indexer = indexer + sampleSize
                # for j in range(0, sampleSize):
                #     if output[j].argmax(axis=0) == target[j]:
                #         accuracyArray[indexer] = 1.0  # Mark with a 1.0 if sample is correctly identified
                #         accuracy = accuracy + 1
                #     indexer = indexer + 1  # update the indexer regardless
        accuracy = accuracy / numSamples
        print("SpikingJelly SNN Accuracy:", accuracy)
        # Clean up the GPU memory
        del currentModel
        torch.cuda.empty_cache()
        return accuracyArray

    # Makes sure the inputs are the right size
    def formatDataLoader(self, dataLoader):
        sampleShape = DMP.GetOutputShape(dataLoader)
        # Check if we need to do resizing, if not just return the original loader
        if sampleShape[1] == self.imgSizeH and sampleShape[2] == self.imgSizeW:
            return dataLoader
        else:  # We need to do resizing
            print("Resize required. Processing now.")
            p = torchvision.transforms.ToPILImage()
            t = torchvision.transforms.ToTensor()
            numSamples = len(dataLoader.dataset)
            sampleShape = DMP.GetOutputShape(dataLoader)  # Get the output shape from the dataloader
            sampleIndex = 0
            batchTracker = 0
            xData = torch.zeros(numSamples, sampleShape[0], self.imgSizeH, self.imgSizeW)
            yData = torch.zeros(numSamples)
            # Go through and process the data in batches...kind of
            for i, (input, target) in enumerate(dataLoader):
                batchSize = input.shape[0]  # Get the number of samples used in each batch
                # print("Resize processing up to=", batchTracker)
                batchTracker = batchTracker + batchSize
                # Save the samples from the batch in a separate tensor
                for batchIndex in range(0, batchSize):
                    # Convert to pil image, resize, convert back to tensor
                    xData[sampleIndex] = t(self.resizeTransform(p(input[batchIndex])))
                    yData[sampleIndex] = target[batchIndex]
                    sampleIndex = sampleIndex + 1  # increment the sample index
            # All the data has been resized, time to put in the dataloader
            newDataLoader = DMP.TensorToDataLoader(xData, yData, transforms=None, batchSize=self.batchSize,
                                                   randomizer=None)
            # Note we don't use the original batch size because the image may have become larger
            # i.e. to large to fit in GPU memory so we use the batch specified in the ModelPlus constructor
            return newDataLoader

    # Go through and delete the main parts that might take up GPU memory
    def clearModel(self):
        print("Warning, model " + self.modelName + " is being deleted and should not be called again!")
        del self.model
        torch.cuda.empty_cache()

        # Only need this for debugging purposes, not used in the transfer chart methods

    def GetCorrectlyIdentifiedSamplesBalanced(self, totalSamplesRequired, dataLoaderInput, numClasses, device):
        # Format the dataloader
        dataLoader = self.formatDataLoader(dataLoaderInput)
        sampleShape = DMP.GetOutputShape(dataLoader)
        xData, yData = DMP.DataLoaderToTensor(dataLoader)
        # Basic error checking
        if totalSamplesRequired % numClasses != 0:
            raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
        # Get the number of samples needed for each class
        numSamplesPerClass = int(totalSamplesRequired / numClasses)
        # correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
        correctlyClassifiedSamples = torch.zeros(((numClasses,) + (numSamplesPerClass,) + sampleShape))
        sanityCounter = torch.zeros((numClasses))
        # yPred = model.predict(xData)
        # yPred = predictD(dataLoader, numClasses, model)
        accuracyArray = self.validateDA(dataLoader, device)
        for i in range(0, xData.shape[0]):  # Go through every sample
            # predictedClass = yPred[i].argmax(axis=0)
            trueClass = yData[i]  # .argmax(axis=0)
            currentSavedCount = int(
                sanityCounter[int(trueClass)])  # Check how may samples we previously saved from this class
            # If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
            if accuracyArray[i] == 1.0 and currentSavedCount < numSamplesPerClass:
                correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i]  # Save the sample
                sanityCounter[int(trueClass)] = sanityCounter[
                                                    int(trueClass)] + 1  # Add one to the count of saved samples for this class
        # Now we have gone through the entire network, make sure we have enough samples
        for c in range(0, numClasses):
            if sanityCounter[c] != numSamplesPerClass:
                raise ValueError("The network does not have enough correctly predicted samples for this class.")
        # Assume we have enough samples now, restore in a properly shaped array
        # xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
        xCorrect = torch.zeros(((totalSamplesRequired,) + sampleShape))
        yCorrect = torch.zeros((totalSamplesRequired))
        currentIndex = 0  # indexing for the final array
        for c in range(0, numClasses):  # Go through each class
            for j in range(0, numSamplesPerClass):  # For each sample in the class store it
                xCorrect[currentIndex] = correctlyClassifiedSamples[c, j]
                yCorrect[currentIndex] = c
                # yCorrect[currentIndex, c] = 1.0
                currentIndex = currentIndex + 1
                # return xCorrect, yCorrect
        cleanDataLoader = DMP.TensorToDataLoader(xCorrect, yCorrect, transforms=None, batchSize=dataLoader.batch_size,
                                                 randomizer=None)
        return cleanDataLoader

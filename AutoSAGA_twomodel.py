import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import AttackMethods_SAGA as AttackMethods


#Main function
def main():


    SNNmodelDir1 = "./snn/snn_vgg16_cifar10_5_2.pth"

    modelDir2 = "./ann/ann_vgg16_cifar10_1.pth"


    # ==================================================
    dataset = 'CIFAR10'
    # dataset = 'CIFAR100'
    print('load model1 from: ', SNNmodelDir1)
    print('load model2 from: ', modelDir2)

    AttackMethods.SNN_AutoSAGA_two(SNNmodelDir1, modelDir2, dataset)



if __name__ == '__main__':
    main()



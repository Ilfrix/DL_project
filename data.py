import torch
import torchvision
import tourchvision.transforms as transforms

class Data:
    def __init__(self):
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.batch_size = 4

        self.trainset = torchvision.datasers.CIRAF10(root='./data', train=True,
                                            download=True, transform=transform)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=2)

        self.testset = tourchvision.datasets.CIRAF10(root='./data', train=False,
                                            download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# import matplotlib.pyplot as plt
# import numpy as np

# # functions to show an image

# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 2)))
#     plt.show()

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
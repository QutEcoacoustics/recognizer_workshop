
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision

from PIL import Image as im



class CNN_4_Layers(nn.Module):

    """
    This is our parameterised 4-layer convolutional network    
    """
    num_classes: int
    embedding_size: int
    num_filters_1: int
    num_filters_2: int
    num_filters_3: int
    num_filters_4: int

    def __init__(self, embedding_size, num_classes, num_filters_1, num_filters_2, num_filters_3, num_filters_4):
        super(CNN_4_Layers, self).__init__()

        # Setting up the network        
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.num_filters_3 = num_filters_3
        self.num_filters_4 = num_filters_4      

        self.conv1 = nn.Conv2d(1, num_filters_1, kernel_size=5)
        self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=5)
        self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=5)
        self.conv4 = nn.Conv2d(num_filters_3, num_filters_4, kernel_size=5)
        
        self.bn1 = nn.BatchNorm2d(num_filters_1)
        self.bn2 = nn.BatchNorm2d(num_filters_2)         
        self.bn3 = nn.BatchNorm2d(num_filters_3)  

        self.fc1 = nn.Linear( num_filters_4 * 4 * 12, embedding_size)
        self.fc2 = nn.Linear( embedding_size, num_classes)

    # Forward pass
    def forward(self, x):
        #print(x.shape)
        x = self.bn1( F.relu(F.max_pool2d(self.conv1(x), 2)) )
         
        #print(x.shape)
        x = self.bn2( F.relu(F.max_pool2d(self.conv2(x), 2)) )
        
        x = self.bn3( F.relu(F.max_pool2d(self.conv3(x), 2)) )
        #print(x.shape)

        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        #print(x.shape)
        
        x = x.view(-1, self.num_filters_4 * 4 * 12)
        self.emb = self.fc1(x)

        x = F.relu(self.emb)
        x = F.dropout(x, 0.5, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x, dim = 1)


def embedding_from_spec_via_network(net: CNN_4_Layers, spec: im):
   
    """
    This returns the embedding layer after passing the given image through the network    
    """
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.3], [0.3])
    ])

    data_tensor = transform(spec).unsqueeze_(0)

    net.eval()
    net(data_tensor)

    return net.emb



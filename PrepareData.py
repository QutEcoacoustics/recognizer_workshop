import torch
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

import numpy as np
import string
import typer
from typing import Optional


# Our includes
import utils
import NeuralNets

import RavenBinaryDataset

# Parameters
log_interval = 50


torch.backends.cudnn.enabled = False
torch.manual_seed(3)


train_losses = []
train_counter = []
test_losses = []
test_counter = []


def test_network(net, test_loader):
    net.eval()
    test_loss = 0
    correct_count = 0
    
    item_count = 0
    correct_items = list(range(0,len(test_loader.dataset)))

    with torch.no_grad():
        for data, target in test_loader:
            #data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #print(pred)
            corr = pred.eq(target.view_as(pred))
            for i, x in enumerate(corr):
                idx = item_count + i
                if x:
                    correct_items[idx] = 1
                else:
                    correct_items[idx] = 0
            item_count += len(corr)        
            
            correct_count += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct_count, len(test_loader.dataset),
        100. * correct_count / len(test_loader.dataset)))

    item_list = test_loader.dataset.get_item_list()
    for i, item in enumerate(item_list):
        print( str(item_list[i]) + "   " + str(correct_items[i]))

    #test_loader
    #item_list = test_loader.d self.item_list

def train_network(epoch, net, optim, train_loader, trainedModelPath):
    '''
    Train the netowrk with the data from train_loader
    '''
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optim.zero_grad()
        output = net(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optim.step()
        if batch_idx % log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        
            train_losses.append( loss.item() )
            train_counter.append( (batch_idx*32) + ((epoch-1)*len(train_loader.dataset)) )
        
    torch.save(net.state_dict(), trainedModelPath)



def train(params, params_spec):
    
    print("Training ...")

    spec_params = {}
    spec_params["fftOverlap"] = 0.5
    spec_params["fftWinSize"] = 512
    spec_params["maxFreq"] = 4000
    spec_params["timeWin"] = 2.0

    class_repetitions = {}
    class_repetitions["pos"] = 10
    class_repetitions["neg"] = 10
 

    transform = transforms.Compose([                   
        transforms.ToTensor(),
        transforms.Normalize([0.3], [0.3])             
    ])

    ds_train, ds_test = RavenBinaryDataset.MakeRavenBinaryDatasetSplit( "data.csv", "data_large.csv", 9135, 12, spec_params, 
        class_repetitions, transform = transform )  

    loader_train = torch.utils.data.DataLoader( ds_train, int(params["batchSize"]), shuffle=True)
    loader_test = torch.utils.data.DataLoader( ds_test, int(params["batchSize"]), shuffle=True)

    
    # This is out standard network architecture
    # network = NeuralNets.CNN_4_Layers(512, 2, 12, 24, 32, 48)

 
    network = NeuralNets.CNN_4_Layers(512, 112, 12, 24, 32, 48)
    network_state_dict = torch.load( "model_synth_base.pth")
    network.load_state_dict(network_state_dict)
    network.fc1 = torch.nn.Linear( 48 * 4 * 12, 512)    
    network.fc2 = torch.nn.Linear( 512, 2)
  	
    #for param in network.parameters():
    #    param.requires_grad = False

    for param in network.fc1.parameters():
        param.requires_grad = True

    for param in network.fc2.parameters():
        param.requires_grad = True

    network.num_classes = 2


    lr = float(params["lr"])
    #lr = 0.001

    print("Learning rate: " + str(lr))

    optimizer = optim.SGD(network.parameters(), lr, momentum=0.9)
    
    for epoch in range( int(params["epochs"])):
        
        train_network(epoch, network, optimizer, loader_train, params["trainedModel"])
        test_network(network, loader_test)
        
        #lrlr = lrlr * float(params["lrDecay"])


def getParamsAndTrain (
    config: Optional[str] = typer.Option("", "-conf"),
    rndSeed: Optional[int] = typer.Option(10, "-rnd-seed"),    
    epochs: Optional[int] = typer.Option(10, "-epochs"),    
    batchSize: Optional[int] = typer.Option(4, "-batch-size"),
    lr: Optional[float] = typer.Option(0.001, "-lr"),    
    lrDecay: Optional[float] = typer.Option(0.098, "-lr-decay"),
    dropOut: Optional[float] = typer.Option(0.5, "-dropout"),    
    dataCsv: Optional[str] = typer.Option(..., "-data-csv"),
    wavDir: Optional[str] = typer.Option(..., "-wav-dir"), 
    specDir: Optional[str] = typer.Option(..., "-spec-dir"),        
    baseModel: Optional[str] = typer.Option("model.pth", "-base-model"),
    trainedModel: Optional[str] = typer.Option("model.pth", "-trained-model")        
    ):

    params = utils.read_config(config, "train")
    params_spec = utils.read_config(config, "spec")
    '''
    This bit of unusual code will put the variables passed into the function
    into the params dictionary, except for the "params" and "params_spec" variables themselves .
    '''   
    for n in getParamsAndTrain.__code__.co_varnames:
        if n != "n" and n != "params" and n != "params_spec" and eval(n) != None:
            params[n] = eval(n)

    utils.print_params(params)    
    utils.print_params(params_spec)   
    
    train(params, params_spec)


if __name__ == "__main__":
    params = typer.run(getParamsAndTrain)









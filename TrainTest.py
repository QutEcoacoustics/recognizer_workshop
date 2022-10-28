
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from datetime import datetime
import NeuralNets
import RavenBinaryDataset
import utils


torch.backends.cudnn.enabled = False
torch.manual_seed(3)


train_losses = []
train_counter = []
test_losses = []
test_counter = []


def test_network(net, test_loader, log):


    # Make sure that the network is in eval mode
    net.eval()

    test_loss = 0
    correct_count = 0
    
    item_count = 0
    correct_items = list(range(0,len(test_loader.dataset)))

    with torch.no_grad():
        for data, target in test_loader:
            # Probably don't need to shift data to device
            #data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            # 
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

    message_str = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct_count, len(test_loader.dataset),
        100. * correct_count / len(test_loader.dataset))

    print(message_str)

    # Log file
    log.write("\nTest results: \n\n")
    item_list = test_loader.dataset.get_item_list()
    for i, item in enumerate(item_list):
        log.write( str(item_list[i]) + "   " + str(correct_items[i]) + "\n")

    log.write("\n" + message_str)
    log.write("\n\n--------------------------\n\n")

    
def train_network(epoch, net, optim, train_loader, trainedModelPath, log):
    '''
    Train the netowrk with the data from train_loader
    '''
    
    # Make sure that we are in train mode
    net.train()

    # Interval for reporting to the screen
    report_interval = 20

    # Loop over batches of training data
    for batch_idx, (data, target) in enumerate(train_loader):
        optim.zero_grad()
        output = net(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optim.step()
        if batch_idx % report_interval == 0:

            message_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item())

            print(message_str)        
            log.write(message_str+"\n")
            log.flush()

            train_losses.append( loss.item() )
            train_counter.append( (batch_idx*32) + ((epoch-1)*len(train_loader.dataset)) )

    # Save the model at the end of each training epoch    
    torch.save(net.state_dict(), trainedModelPath)



def train(train_params, spec_params):
    """
    Enty point for training the neural network.

    """
    
    print("\n\nTraining ...")

    # Start the log file with parameter values
    log = open( train_params["log"], 'w')
    log.write( "\n-----------\nParameters\n-----------\n\n" )
    log.write( utils.params_to_string(spec_params) )
    log.write( "\n\n" )
    log.write( utils.params_to_string(train_params) )
    log.write( "\n\n" )    
    log.write( "\n-----------\nTraining\n-----------\n" )

    # Write the current time to the log file
    now = datetime.now()    
    log.write( "\nTime:  " + now.strftime("%H:%M:%S") + "\n\n")


    # This map defines the number of times to duplicate data for the train and test sets
    class_repetitions = {}
    class_repetitions["pos"] = 10
    class_repetitions["neg"] = 10
 
    # This is our standard transform
    transform = transforms.Compose([                   
        transforms.ToTensor(),
        transforms.Normalize([0.3], [0.3])             
    ])

    test_set_size = 12
    random_seed = 9135
    # Get the train and test datasets
    ds_train, ds_test = RavenBinaryDataset.MakeRavenBinaryDatasetSplit( train_params["dataCSV"], "data_training.csv", random_seed, test_set_size, spec_params, 
        class_repetitions, transform = transform )  

    # Make the data loaders
    loader_train = torch.utils.data.DataLoader( ds_train, int(train_params["batchSize"]), shuffle=True)
    loader_test = torch.utils.data.DataLoader( ds_test, int(train_params["batchSize"]), shuffle=True)

    network = None
    
    # Load the neural network model if we are using a pre-trained base model
    if not train_params["baseModel"] == "":
        network = NeuralNets.CNN_4_Layers(512, 112, 12, 24, 32, 48)
        network_state_dict = torch.load( train_params["baseModel"] )
        network.load_state_dict(network_state_dict)
        
        # Replace the last 2 layers
        network.fc1 = torch.nn.Linear( 48 * 4 * 12, 512)    
        network.fc2 = torch.nn.Linear( 512, 2)
    
        network.num_classes = 2
    # Use neural network without pre-trained model    
    else:
        network = NeuralNets.CNN_4_Layers(512, 2, 12, 24, 32, 48)
  	
    lr = float(train_params["lr"])    

    # We could expose the momentum parameter as a configuration parameter
    # but don't want to make things too complicated for the moment.
    optimizer = optim.SGD(network.parameters(), lr, momentum=0.9)
    
    # Main training loop
    for epoch in range( int(train_params["epochs"])):
        
        train_network(epoch, network, optimizer, loader_train, train_params["trainedModel"], log)
        test_network(network, loader_test, log)
        








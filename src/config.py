import configparser
import utils



def read_config(file_name: str, section: str):
    """
    Reads the values in the given section of a config file 
    """
    if file_name is not None:  
        config = configparser.ConfigParser()
        config.read(file_name)

        if section not in config.sections():
            print(f'invalid config section provided: "{section}"')
            return {}        

        params = dict(config.items(section))
        params = utils.normalize_map_key_names(params)
        return params
    else:
        return {}

    
def read_spec_params(file_name: str):
    """
    This is a more specific version of the "read_config" function 
    """
    if file_name is not None:  
        config = configparser.ConfigParser()
        config.read(file_name)
        
        params = {}       
        params["maxFreq"] = int(config["spec"]["max-freq"])
        params["timeWin"] = float(config["spec"]["time-win"])
        params["fftWinSize"] = int(config["spec"]["fft-win-size"])

        # Keep this fixed for the moment        
        params["fftOverlap"] = 0.5
        
        return params
    else:
        return {}
    
    
def read_train_params(file_name: str):
    """
    This is a more specific version of the "read_config" function 
    """    
    if file_name != None:  
        config = configparser.ConfigParser()
        config.read(file_name)
        
        params = {}       
        params["dataCSV"] = config["train"]["data-csv"]
        params["epochs"] = int(config["train"]["epochs"])
        params["lr"] = float(config["train"]["lr"])        
        params["batchSize"] = int(config["train"]["batch-size"])
        params["testSetSize"] = int(config["train"]["test-set-size"]) 
        params["baseModel"] = config["train"]["base-model"]        
        params["trainedModel"] = config["train"]["trained-model"]
        params["log"] = config["train"]["log"]
        params["saveImagePatches"] = config["train"]["save-image-patches"] 
        params["randomSeed"] = int(config["train"]["random-seed"])         
        return params
    else:
        return {}    
    

def read_infer_params(file_name: str):
    """
    This is a more specific version of the "read_config" function 
    """

    if file_name is not None:  
        config = configparser.ConfigParser()
        config.read(file_name)
        
        section = "infer"
        
        params = {}       
        params["hop"] = float(config[section]["hop"])
        params["model"] = config[section]["model"]
        params["speciesName"] = config[section]["species-name"]          
        params["recursive"] = config[section]["recursive"]
        params["imageDir"] = config[section]["image-dir"] 
        params["outputDir"] = config[section]["output-dir"]
        params["maxFileImages"] = int(config[section]["max-file-images"]) 
        params["log"] = config[section]["log"]         
        
        return params
    else:
        return {} 
    

def print_params(params):
    """
    Prints the keys and values in a map
    """
    for k, v in params.items():
        print("" + k + " : " + str(v)) 
        

def params_to_string(params):
    """
    Constructs a string from the keys and values in a map
    """
    str_value = ""
    for k,v in params.items():
        str_value = str_value + k + " : " + str(v) + "\n" 

    return str_value
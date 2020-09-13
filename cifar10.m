%{
  Function to load data from CIFAR10.                          
                                                                    
    Parameters                                                      
    ----------                                                      
    data_dir : string                                               
        Absolute path to the directory containing the extracted CIFAR10 files.

    split : string                                              
        Either "train" or "test", which loads the entire train/test data in
        concatenated form.

    Returns
    -------                                                         
    data : ndarray (uint8)                                          
        Data from the CIFAR10 dataset corresponding to the train/test        
        split. The datata should be in NHWC format.                 

    labels : ndarray (int)                                          
        Labels for each data. Integers ranging between 0 and 9.     
                                                                    
    
%}
function [X,Y] = cifar10(data_dir, split)
    
    %Ensure we are passed a valid data dir
    assert(exist(data_dir,'dir') == 7); 
 
    X = uint8.empty(0,32,32,3);
    Y = uint8.empty(0,1);
    if(split == "train")
        for i = 1:5
            [data,y] = load_batch(fullfile(data_dir,['data_batch_' num2str(i) '.mat']));
            X = cat(1, data, X);
            Y = cat(1, y, Y);
        end
    elseif (split == "test")
        [X,Y] = load_batch(fullfile(data_dir,['test_batch.mat']));
    else
        throw(MException('cifar10:invalidDataSplit', ...
        'Argument `split` must be `train` or `test`: %s provided', split));
    end
end

function [data, labels] = load_batch(fullInputBatchPath)
    load(fullInputBatchPath);
    data = data'; %#ok<NODEF>
    data = reshape(data, 32,32,3,[]); %Reshape flat data to image shape
    data = permute(data, [4 1 2 3]); %Re-order so the batch on dim=0
end
%{
    Check if we have already trained a model
    If yes, check how good our model is at infereing the correct label for
    our test dataset
    If no, then train a model
%}

if(~exist(saved_model, 'file'))
    disp("Training a new model");
    [X, Y] = get_features(path_to_cifar, min_max_norm, unit_vector_norm, true); 
    model = train_model(X, Y, saved_model);
else
    disp(["Loading previously saved model:", saved_model]);
    mat = matfile(saved_model);
    model = mat.trained_model;
    [X, Y] = get_features(path_to_cifar, min_max_norm, unit_vector_norm, false);
end

disp("Evaluating the results");
metrics = evaluate(model, X, Y, verbose);

k = keys(metrics);
val = values(metrics);
for i = 1:length(metrics)
     disp(k{i});
     disp(val{i}');
end

function [x_data,y_data] = get_features(path_to_cifar, min_max_norm, unit_vector_norm, train)
    if train 
        %Load cifar10 train data and labels
        disp("Reading training data...");
        [x_data, y_data] = cifar10(path_to_cifar, "train");
    else
        %Load cifar10 test data and labels
        disp("Reading testing data...");
        [x_data, y_data] = cifar10(path_to_cifar, "test");
    end
    
    %Make sure we have the same size of images and labels
    n_data_x = size(x_data, 1);
    n_data_y = size(y_data, 1);
    n_data = n_data_x;
    
    assert(n_data_x == n_data_y);
    disp(["Num of samples:", n_data]); 
    
    %Convert to grayscale and 0-1 float
    x_ = zeros(n_data, 32, 32);
    for i = 1:n_data
        %Convert image to grayscale
        x_gray = double(rgb2gray(squeeze(x_data(i, :, :, :))));
        x_(1, :, :) = x_gray ./ 255;    
    end
    x_data = x_;
    
    disp("Extract HOG features, go grab a coffee ...");
    tic
    
    x_data = hog(x_data);
    toc
            
    if (min_max_norm == true)
       disp("Normalize the HOG features by rescaling the range to [-1, 1]");
       x_data = rescale(x_data, -1, 1);
    elseif (unit_vector_norm == true)
       disp("Normalize the HOG features by converting them to unit vectors");
       x_data = x_data./norm(x_data);
    end
    
    if train
        %Randomly shuffle the training data
        rand_idx = randperm(n_data);
        x_data = x_data(rand_idx, :);
        y_data = y_data(rand_idx);
    end
end    
    
function trained_model = train_model(x_train, y_train, saved_model)
    disp("Training the SVM classifier ... go grab another coffee");
    tic
    t = templateSVM('KernelFunction', 'linear');
    trained_model = fitcecoc(x_train,y_train, 'Learners', t);
    toc
    
    disp(["Saving the model as", saved_model]);
    save(saved_model, 'trained_model', '-v7.3');
end 



function [metrics] = evaluate(model, X, y, verbose)
    eps = 1e-6;

    disp("Use our trained model to predict h: X -> [0,10]");
    pred = predict(model, X);
    
   
    disp("Creating confusion matrix"); 
    cm = confusionmat(y,pred);
   
     
    disp("Calculating the evaluation metrics");
    recall    = diag(cm) ./(sum(cm, 2) + eps);
    precision = diag(cm) ./(sum(cm, 1)' + eps);
    
    avg_recall = mean2(recall);
    avg_precision = mean2(precision);
    fscore = 2 * avg_precision * avg_recall / (avg_precision + avg_recall);
    accuracy = mean2(pred==y);
    
    metrics = containers.Map({'recall', 'precision', 'avg_precision', 'avg_recall', 'fscore', 'accuracy'}, ... 
                             {recall, precision, avg_precision, avg_recall, fscore, accuracy});

    if (verbose == true)
        gcm = plotconfusion(y,pred, 'Confusion Matrix');
        saveas(gcm, 'confusion_matrix.fig');
    end
     
end

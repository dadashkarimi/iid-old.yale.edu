
%% MNIST feature learning demo
% Author: Sebastian Tschiatschek (ETHZ)

clear all;
clc;
clf;
close all;

twoLayer = 1; %set 1 for one hidden layer, 0 for no hidden layer


%% settings
nrHiddens = 100;
nrMaxIter = 80;

%% load data
load('mnist_all.mat');

% compile training and test data into a single matrix/vector
nrTest = 0;
nrTrain = 0;
for i = 1:10,
    nrTest = nrTest + eval(sprintf('size(test%d,1);', i-1));
    nrTrain = nrTrain + eval(sprintf('size(train%d,1);', i-1));
end

inputTest = zeros(nrTest, 28*28);
outputTest = zeros(nrTest, 1);
inputTrain = zeros(nrTrain, 28*28);
outputTrain = zeros(nrTrain, 1);
offsetTest = 1;
offsetTrain = 1;
for i = 1:10,
    nrTest = eval(sprintf('size(test%d,1);', i-1));
    nrTrain = eval(sprintf('size(train%d,1);', i-1));
    
    eval(sprintf('inputTest(offsetTest:(offsetTest+nrTest-1),:) = test%d;', i-1));
    eval(sprintf('outputTest(offsetTest:(offsetTest+nrTest-1),:) = %d;', i));
    eval(sprintf('inputTrain(offsetTrain:(offsetTrain+nrTrain-1),:) = train%d;', i-1));
    eval(sprintf('outputTrain(offsetTrain:(offsetTrain+nrTrain-1),:) = %d;', i));
    
    offsetTest = offsetTest + nrTest;
    offsetTrain = offsetTrain + nrTrain;
end

% randomly permute training data
perm = randperm(size(inputTrain,1));
inputTrain = inputTrain(perm,:);
outputTrain = outputTrain(perm,:);

%% show some images
for r = 1:10,
    for c = 1:10,
        idx = (r-1)*10+c;
        subplot(10,10,idx);
        imagesc(reshape(inputTrain(idx,:), 28, 28)');
        set(gca, 'XTickLabel', []); set(gca, 'YTickLabel', []);
        colormap(gray);
    end
end
pause

%% reduce data
perm = perm(1:10000); % only use subset of samples
inputTrain = inputTrain(perm,:);
outputTrain = outputTrain(perm,:);

%% set up neural network and train it (show learned filters);
if twoLayer,
    net = network;
    net.numInputs = 1;
    net.numLayers = 2;
    net.inputs{1}.size = 28*28;
    net.layers{1}.size = nrHiddens;
    net.layers{1}.transferFcn = 'tansig';
    net.layers{1}.initFcn = 'initnw';
    net.layers{2}.size = 10;
    net.layers{2}.transferFcn = 'softmax';
    net.layers{2}.initFcn = 'initnw';
    net.biasConnect(1) = 1;
    net.biasConnect(2) = 1;
    net.outputConnect = [0 1];
    net.inputConnect(1,1) = 1;
    net.layerConnect(2,1) = 1;
    net.performFcn = 'crossentropy'; % 'crossentropy';
    net.trainFcn = 'traingdx';
    net.plotFcns = {'plotperform','plottrainstate'};
    net.inputs{1}.exampleInput = inputTrain';
    net.initFcn = 'initlay';
    % net.trainParam.lr = 1e-9;
    net.trainParam.epochs = 100;
    % net.trainParam.min_grad = 1e-9;
    net.performParam.regularization = 1e-3;
    % net.divideFcn = 'dividerand';
    net.inputs{1}.processFcns = {'mapstd'};
    net = init(net);
    
    % good initialization
    net.iw{1} = 1e-3*randn(size(net.iw{1}));
    net.iw{2} = 1e-3*randn(size(net.iw{2}));
    
    % bad initialization
    % net.iw{1} = 1e1*randn(size(net.iw{1}));
    % net.iw{2} = 1e1*randn(size(net.iw{2}));
else
    net = network;
    net.numInputs = 1;
    net.numLayers = 1;
    net.inputs{1}.size = 28*28;
    net.layers{1}.size = 10;
    net.layers{1}.transferFcn = 'softmax';
    net.layers{1}.initFcn = 'initnw';
    net.biasConnect(1) = 1;
    net.outputConnect = [1];
    net.inputConnect(1,1) = 1;
    %net.layerConnect(2,1) = 1;
    net.performFcn = 'crossentropy'; % 'crossentropy';
    net.trainFcn = 'traingdm';
    net.plotFcns = {'plotperform','plottrainstate'};
    net.inputs{1}.exampleInput = inputTrain';
    net.initFcn = 'initlay';
    net.trainParam.epochs = 100;
    net.trainParam.min_grad = 1e-6;
    %net.performParam.regularization = 1e-3;
    %net.divideFcn = 'dividerand';
    net.inputs{1}.processFcns = {'mapstd'};
    net = init(net);
    net.iw{1} = 1e-3*randn(size(net.iw{1})); % good initialization
end

% show weights
outputTrainMatrix = zeros(length(outputTrain), 10);
idx = sub2ind(size(outputTrainMatrix), (1:length(outputTrain))', outputTrain(:));
outputTrainMatrix(idx) = 1;

% net = train(net, inputTrain', outputTrainMatrix');

figure;
for iter = 1:1,
    net = train(net, inputTrain', outputTrainMatrix');
    
    % train
    pred = sim(net, inputTrain');
    [a,cl] = max(pred,[],1);
    fprintf(1, 'Training accuracy = %f\n', mean(cl(:) == outputTrain(:)));

    % test
    pred = sim(net, inputTest');
    [a,cl] = max(pred,[],1);
    fprintf(1, 'Test accuracy = %f\n', mean(cl(:) == outputTest(:)));
    
    for r = 1:2,
        for c = 1:5,
            idx = (r-1)*5+c;
            subplot(2,5,idx);
            imagesc(reshape(net.IW{1}(idx,:), 28, 28)');
            set(gca, 'XTickLabel', []); set(gca, 'YTickLabel', []);
            colormap(gray);
        end
    end
    drawnow;
    
    if iter == 2,
        pause;
    end
end

% train
pred = sim(net, inputTrain');
[a,cl] = max(pred,[],1);
fprintf(1, 'Training accuracy = %f\n', mean(cl(:) == outputTrain(:)));

% test
pred = sim(net, inputTest');
[a,cl] = max(pred,[],1);
fprintf(1, 'Test accuracy = %f\n', mean(cl(:) == outputTest(:)));

figure;
for r = 1:2,
    for c = 1:5,
        idx = (r-1)*5+c;
        subplot(2,5,idx);
        imagesc(reshape(net.IW{1}(idx,:), 28, 28)');
        colormap(gray);
    end
end
%pause

%% fitting a sinusoid with noise: Neural Nets vs Kernels
% Author: Sebastian Tschiatschek (ETHZ)

clear all;
clf;
clc;

noise_sigma=0.01;

%% settings
range = -6:0.01:6;

% function to approximate
f = @(x,sigma) sin(x) + sigma*randn(size(x));



% generate data
x = -6:0.2:6;
y = f(x, noise_sigma);

%% plot
subplot(3,1,1);
plot(range, f(range, 0));
hold on;
scatter(x, y);
title('function we want to approximate');
pause

%% approximate using an increasing number of hidden units
nrHiddensRange = [1,5,10,20,50];
labels = {};
for i = 1:length(nrHiddensRange),
    labels{i} = sprintf('%d', nrHiddensRange(i));
end

errors = NaN(length(nrHiddensRange), 1);
for idx = 1:length(nrHiddensRange),
    nrHiddens = nrHiddensRange(idx);
    
    net = network;
    net.numInputs = 1;
    net.numLayers = 2;
    net.inputs{1}.size = 1;
    net.layers{1}.size = nrHiddens;
    net.layers{1}.transferFcn = 'tansig';
    net.layers{1}.initFcn = 'initnw';
    net.layers{2}.size = 1;
    net.layers{2}.transferFcn = 'purelin';
    net.layers{2}.initFcn = 'initnw';
    net.biasConnect(1) = 1;
    net.outputConnect = [0 1];
    net.inputConnect(1,1) = 1;
    net.layerConnect(2,1) = 1;
    net.performFcn = 'mse';
    net.trainFcn = 'trainlm';
    net.plotFcns = {'plotperform','plottrainstate'};
    net.inputs{1}.exampleInput = x;
    net.initFcn = 'initlay';
    net.trainParam.epochs = 100;
    net.trainParam.min_grad = 1e-4;
    
    % use earlystopping
    net.divideFcn = 'dividerand';
    
    net = init(net);
    
    % "good" initialization
    % net.iw{1} = 1e-3*randn(size(net.iw{1}));
    
    % bad initialization
    net.iw{1} = 1e1*randn(size(net.iw{1}));

    % train
    net = train(net,x,y);
    norm(net.iw{1})
    approx = sim(net,range);
    
    % plot
    plt2 = subplot(3,1,2);
    hold off;
    plot(range, f(range, 0), 'g', 'LineWidth', 4);
    hold on;
    scatter(x, y);
    hold on;
    plot(range,approx, 'r', 'LineWidth', 2);
    title(sprintf('%d hidden units', nrHiddens));
    
    pause
end

%% now compare against Kernel ridge regression
n = length(x);
xaug = [x', ones(n,1)];
scale_par = 3.5;
bias_par = 0;

Ktr = tanh(xaug*xaug'/scale_par-bias_par);
alpha = (Ktr+noise_sigma*eye(n))\(y');
ypr = alpha'*Ktr;
subplot(3,1,3);
scatter(x,y); hold on;
title('kernel logistic regression fit with neural network kernel');
plot(x,ypr)
hold off
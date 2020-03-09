%% illustration of the universal approximator theorem
% Author: Sebastian Tschiatschek (ETHZ)

clear all;
clf;
clc;

%% settings
range = -6:0.01:6;

% function to approximate
%f = @(x) sin(x).*exp(abs(x)); % f = @(x) sin(x).*floor(abs(x));
f = @(x) sin(x.*floor(abs(x)));

%% plot
subplot(3,1,1);
plot(range, f(range));
title('function we want to approximate');

%% sample a subset of points
pos = min(range) + (max(range) - min(range))*rand(1000,1);
pos = sort(pos);
vals = f(pos);
plt2 = subplot(3,1,2);
scatter(pos, vals);
title('sampled points');
pause

%% approximate using an increasing number of hidden units
nrHiddensRange = [1,2,5,10,20,30,40,50,200];
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
    net.inputs{1}.exampleInput = pos';
    net.initFcn = 'initlay';
    net.trainParam.epochs = 100;
    net.trainParam.min_grad = 1e-4;
    net = init(net);


    % train
    net = train(net,pos',vals');
    approx = sim(net,range);
    errors(idx) = mean((approx-f(range)).^2);
    
    % plot
    plt2 = subplot(3,1,2);
    hold off;
    plot(range, f(range), 'g', 'LineWidth', 4);
    hold on;
    plot(range,approx, 'r', 'LineWidth', 2);
    title(sprintf('%d hidden units', nrHiddens));
    
    subplot(3,1,3);
    bar(1:length(nrHiddensRange), log(errors));
    set(gca, 'xticklabel', labels);
    xlabel('# hidden units');
    ylabel('log MSE');
    pause
end
% Demo of the cost-sensitive linear SVM
% Author: Andreas Krause (ETHZ)

function cost_sensitive_svm_demo

rng(0)

use_std = .7; % inject noise

% prepare data
gridsize = .02;
npos = 20;  % number of positive examples
nneg = 100; % number of negativeexamples
n = npos+nneg;
X(1:npos,:)=randn(npos,2)*use_std+1;
Y(1:npos)=1;
X((npos+1):n,:)=randn(nneg,2)*use_std;
Y((npos+1):n)=-1;
X(:,3)=1;
minX = min(X(:,1))-.3; maxX = max(X(:,1))+.3;
minY = min(X(:,2))-.3; maxY = max(X(:,2))+.3;

multipliers=[.01 .1 1 10 100]; % mislabeling cost ratios
lambda = .001; % regularization parameter

for i = 1:length(multipliers) %visualize solution for each ratio
    clf
    % train SVM with particular cost ratio
    w = train_svm_with_sgd(X,Y,[multipliers(i) 1],lambda);
    
    % make predictions on mesh for plotting
    [Xg,Yg] = meshgrid(minX:gridsize:maxX,minY:gridsize:maxY);
    fval = sign([Xg(:) Yg(:) ones(length(Xg(:)),1)]*w);
    
    % some plotting
    clf
    axis([minX maxX minY maxY])
    hold on
    plot(X(1:npos,1),X(1:npos,2),'*');
    plot(X((npos+1):n,1),X((npos+1):n,2),'s');
    contour(Xg,Yg,reshape(fval,size(Xg)));
    hold off
    title(sprintf('Mislabeling cost ratio: %0.2f',multipliers(i)))
    pause
end
    
function w = train_svm_with_sgd(X,Y,classMult,lambda)
w = rand(3,1); %coefficients
n = size(X,1);
niter = 10000;
for i = 1:niter

    j = randi(n,1); %pick random data point
    
    eta = min(1,100/i);
    if Y(j)*X(j,:)*w < 1 % violate margin
        % gradient step, learning rate = eta, 
        % cost-sensitive hinge loss
        w = w+Y(j)*X(j,:)'*eta*classMult((-Y(j)+3)/2); %(y+3)/2 maps {-1,1} to {1,2} for indexing
        w = w*(1-lambda*eta);
    end
    
end
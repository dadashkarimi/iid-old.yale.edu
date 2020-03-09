% Demo of the SVM with L1-regularizer
% Author: Andreas Krause (ETHZ)

function l1_svm_demo

rng(0)

use_std = .3; % inject noise

% prepare data
gridsize = .02;
npos = 50;  % number of positive examples
nneg = 50; % number of negativeexamples
n = npos+nneg;
X(1:npos,:)=randn(npos,2)*use_std;
Y(1:npos)=1;
X(1:npos,1)=X(1:npos,1)+.5;
X(1:npos,2)=X(1:npos,2)+.1;
X((npos+1):n,:)=randn(nneg,2)*use_std;
X((npos+1):n,1)=X((npos+1):n,1)-.5;
X((npos+1):n,2)=X((npos+1):n,2)-.1;
Y((npos+1):n)=-1;
X(:,3)=1;
minX = min(X(:,1))-.3; maxX = max(X(:,1))+.3;
minY = min(X(:,2))-.3; maxY = max(X(:,2))+.3;

lambdas=[.001 .003 .01 .03 .1 .3 1]; % mislabeling cost ratios

for i = 1:length(lambdas) %visualize solution for each ratio
    clf
    % train perceptron with particular cost ratio
    w = train_svm_with_sgd(X,Y,lambdas(i))
    
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
    title(sprintf('Regularization parameter: %0.3f',lambdas(i)))
    pause
end
    
function w = train_svm_with_sgd(X,Y,lambda)
w = rand(3,1); %coefficients
n = size(X,1);
niter = 10000;
for i = 1:niter

    j = randi(n,1); %pick random data point
    
    eta = min(1,100/i);
    if Y(j)*X(j,:)*w < 1 % violate margin
        % gradient step, learning rate = eta, 
        % cost-sensitive hinge loss
        w = w+Y(j)*X(j,:)'*eta; %(y+3)/2 maps {-1,1} to {1,2} for indexing
        w(1:(length(w)-1)) = sign(w(1:(length(w)-1))).*max(abs(w(1:(length(w)-1)))-eta*lambda,0); % apply shrinkage (proximal) operator 
    end
    
end
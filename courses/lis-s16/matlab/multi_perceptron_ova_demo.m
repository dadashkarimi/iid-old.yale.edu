% Demo of multiclass prediction with linear Perceptron 
% Author: Andreas Krause (ETHZ)

function multi_perceptron_ova_demo

use_std = .1; % inject noise
npc = [50 50 50]; % how many classes / points per class?


% prepare multiclass data
gridsize = .02;
numclasses = length(npc);
symbols = '.ox+*sdphv^<>'; %for labeling data points on the plot
n = sum(npc);
X = [];
Y = [];
for c = 1:numclasses
    nc = npc(c);
    meanc = [randn(1,2) 1];
    Xc = ones(nc,1)*meanc;
    Xc(:,1:2) = Xc(:,1:2)+randn(nc,2)*use_std;
    Yc = ones(nc,1)*c;
    X = [X; Xc];
    Y = [Y; Yc];
end
minX = min(X(:,1))-.3; maxX = max(X(:,1))+.3;
minY = min(X(:,2))-.3; maxY = max(X(:,2))+.3;

W = zeros(numclasses,3);
for c = 1:numclasses %loop over all classes
    % prepare data for classifying c against rest
    posind = (Y==c); negind = (Y~=c);
    Yova = ones(n,1); %set labels for one-vs-all classification
    Yova(negind)=-1;

    % train perceptron for class c vs. rest
    W(c,:) = train_perceptron_with_sgd(X,Yova)';
    W(c,:) = W(c,:)/norm(W(c,:));
    
end

% find confidence for all classifiers
[Xg,Yg] = meshgrid(minX:gridsize:maxX,minY:gridsize:maxY);
fval = ([Xg(:) Yg(:) ones(length(Xg(:)),1)]*(W'));
[~,argmaxfval]=max(fval');
argmaxfval = reshape(argmaxfval,size(Xg));

% now plot decision region
hold on; axis([minX maxX minY maxY])
h = pcolor(Xg,Yg,argmaxfval);
set(h,'linestyle','none');

for c = 1:numclasses 
    posind = (Y==c);  Xpos = X(posind,:);
    % plot data points from class c
    plot(Xpos(:,1),Xpos(:,2),symbols(c));

    % make predictions on mesh for plotting decision boundaries
    [Xg,Yg] = meshgrid(minX:gridsize:maxX,minY:gridsize:maxY);
    fval = sign([Xg(:) Yg(:) ones(length(Xg(:)),1)]*(W(c,:)'));
    plot(Xpos(:,1),Xpos(:,2),symbols(c));
    contour(Xg,Yg,reshape(fval,size(Xg)));

end


    
function w = train_perceptron_with_sgd(X,Y)
w = rand(3,1); %coefficients
n = size(X,1);
niter = 10000;
for i = 1:niter

    j = randi(n,1); %pick random data point
    
    eta = min(1,100/i);
    if Y(j)*X(j,:)*w < 0 % misclassified
        % gradient step, learning rate = eta, 
        % cost-sensitive Perceptron loss
        w = w+Y(j)*X(j,:)'*eta; %(y+3)/2 maps {-1,1} to {1,2} for indexing
    end
    
end
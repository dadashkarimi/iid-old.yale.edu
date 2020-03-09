% Demo of the linear SVM trained via SGD
% Author: Andreas Krause (ETHZ)

clf; clear;

lambda = .01; %regularization parameter
use_std = .3; %noise injected

%prepare data
gridsize = .02;

npc = 20; % points per class
n = 2*npc;
X(1:npc,:)=randn(npc,2)*use_std+1;
Y(1:npc)=1;
X((npc+1):n,:)=randn(npc,2)*use_std;
Y((npc+1):n)=-1;
minX = min(X(:,1)); maxX = max(X(:,1));
minY = min(X(:,2)); maxY = max(X(:,2));


w = rand(3,1); %coefficients
niter = 100;
pause
for i = 1:niter
    % predict discriminant for plotting
    [Xg,Yg] = meshgrid(minX:gridsize:maxX,minY:gridsize:maxY);
    fval = sign([Xg(:) Yg(:) ones(length(Xg(:)),1)]*w);

    j = randi(n,1); %pick random data point
    fpred = ([X(j,:),1]*w);
    
    % some plotting
    clf
    axis([minX maxX minY maxY]);
    hold on
    plot(X(1:npc,1),X(1:npc,2),'o');
    plot(X((npc+1):n,1),X((npc+1):n,2),'s');
    contour(Xg,Yg,reshape(fval,size(Xg)));
    plot(X(j,1),X(j,2),'ks','markersize',10);
    hold off
    pause(.1)
    
    % Stochastic gradient descent step
    eta = min(5/i,1); %learning rate
    if Y(j)*fpred<1 %violate margin
        w = w+eta*Y(j)*[X(j,:),1]'; %gradient w.r.t. hinge loss
    end
    w = w-eta*lambda*w; %gradient w.r.t. regularizer
    
end
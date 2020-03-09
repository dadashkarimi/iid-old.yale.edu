% Demo for a kernelized Perceptron trained via SGD
% by Andreas Krause (ETHZ)

clf; clear;


%
use_std=.1; %amount of noise

% prepare the data
bw=.5;
gridsize = .02;
npc = 50;
n = 2*npc;
alpha = zeros(1,n);
X(1:npc,:)=randn(npc,2);
X(1:npc,:)=X(1:npc,:)./(sqrt(sum(X(1:npc,:).^2,2))*ones(1,2));
X(1:npc,:)=X(1:npc,:)+randn(npc,2)*use_std;
Y(1:npc)=1;
X((npc+1):n,:)=randn(npc,2)*use_std;
Y((npc+1):n)=-1;
minX = min(X(:,1)); maxX = max(X(:,1));
minY = min(X(:,2)); maxY = max(X(:,2));


%compute kernel matrix
K = exp(-dist(X').^2/bw^2);
niter = 50;
for i = 1:niter
    % make predictions for plotting
    [Xg,Yg] = meshgrid(minX:gridsize:maxX,minY:gridsize:maxY);
    Kpred = exp(-dist(X,[Xg(:),Yg(:)]').^2/bw^2);
    fval = ((alpha*Kpred));

    j = randi(n,1); %pick random data point
    ypred = sign(alpha*K(:,j)); %predict label
    
    % plot decision boundary
    clf
    axis([minX maxX minY maxY]);
    hold on
    plot(X(1:npc,1),X(1:npc,2),'o');
    plot(X((npc+1):n,1),X((npc+1):n,2),'s');
    contour(Xg,Yg,reshape(fval,size(Xg)),10);
    contour(Xg,Yg,reshape(sign(fval),size(Xg)),10);
    plot(X(j,1),X(j,2),'ks','markersize',10);
    hold off
    pause
    
    if ypred~=Y(j) % incorrect prediction
        alpha(j)=alpha(j)+Y(j); %learning rate = 1
    end
    
end
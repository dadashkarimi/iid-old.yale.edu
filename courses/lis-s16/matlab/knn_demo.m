% Demo for k-NN classification
% by Andreas Krause (ETHZ)

clear; clf;

use_std=0.1; %amount of noise
K = 3; % number of neighbors to use


% prepare the data
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

% make predictions on grid
[Xg,Yg] = meshgrid(minX:gridsize:maxX,minY:gridsize:maxY);
% compute all distances
dists = dist(X,[Xg(:),Yg(:)]');
[~,sdists] = sort(dists); % sort
fval = zeros(length(Xg(:)),1);
for j = 1:length(Xg(:))
    fval(j)=sum(Y(sdists(1:K,j)))/K; %average predictions of K nearest points
end

clf
hold on
plot(X(1:npc,1),X(1:npc,2),'o');
plot(X((npc+1):n,1),X((npc+1):n,2),'s');
contour(Xg,Yg,reshape(sign(fval),size(Xg)),10);
hold off


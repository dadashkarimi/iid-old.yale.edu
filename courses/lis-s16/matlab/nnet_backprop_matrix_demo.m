% Demo for training neural nets via backpropagation
% Hidden layer: rectified linear units
% Output layer: hinge loss
% by Andreas Krause (ETHZ)

function W=nnet_backprop_matrix_demo
npc = 100; %points per class
dataset = 2; %choose between 1 and 2
use_std=.2; %amount of noise
if dataset == 1 %linearly separable
    X = [rand(npc,2); -rand(npc,2)]; 
    X(:,3)=ones(2*npc,1);
    Y = (-sign(X(:,1)));
elseif dataset == 2 %nonlinear boundary
    X(1:npc,:)=randn(npc,2);
    X(1:npc,:)=X(1:npc,:)./(sqrt(sum(X(1:npc,:).^2,2))*ones(1,2));
    X(1:npc,:)=X(1:npc,:)+randn(npc,2)*use_std;
    Y(1:npc,1)=-1;
    X((npc+1):(2*npc),:)=randn(npc,2)*use_std;
    X(:,3)=ones(2*npc,1);
    Y((npc+1):(2*npc),1)=1;
else 
    error('Data set index must be 1 or 2');
end

nunits = [1 20 size(X,2)]; %specifies no. units per layer; last is #inputs
niter = 10000;

W = backprop_train(X,Y,nunits,niter); %train neural net via SGD

%do some plotting
plot(X(Y==1,1),X(Y==1,2),'ks');
hold on
plot(X(Y==-1,1),X(Y==-1,2),'ro');
minX = min(X(:,1)); maxX = max(X(:,1));
minY = min(X(:,2)); maxY = max(X(:,2));
gridsize=.02;
[Xg,Yg] = meshgrid(minX:gridsize:maxX,minY:gridsize:maxY);
fval = backprop_predict([Xg(:) Yg(:) ones(length(Xg(:)),1)],W);
contour(Xg,Yg,reshape(fval,size(Xg)));
contour(Xg,Yg,reshape(sign(fval+1e-10),size(Xg)),'linewidth',2);
hold off






function W = backprop_train(X,Y,nunits,niter)

nlayers = length(nunits)-1;
W = {};
for i = 1:nlayers
    W{i} = randn(nunits(i),nunits(i+1))/sqrt(nunits(i+1));
end

for i = 1:niter
    % SGD; pick random data point
    dataind = randi(size(X,1));
    x = X(dataind,:)';
    y = Y(dataind);
    % forward propagation
    v={}; z={};
    v{nlayers+1}=x;
    for l = (nlayers):-1:2
        z{l}=W{l}*v{l+1};
        v{l}=nonlin(z{l});
    end
    v{1}=W{1}*v{2}; %output layer applies no nonlinearity
    % backpropagation
    dV = {}; %partial derivatives wrt units
    dW = {}; %partial derivatives wrt weights
    delta = {}; %error signals
    % output layer handled separately due since loss != activation fn
    dV{1} = lossgrad(v{1},y); %loss derivative at output unit
    delta{1} = dV{1};
    dW{1} = delta{1}*v{2}'; 
    % now the hidden layers
    for l = 2:nlayers
        dV{l} = nonlingrad(z{l}); %partial deriv. of activation fn.
        delta{l} = dV{l}.*(W{l-1}'*delta{l-1});
        dW{l} = delta{l}*v{l+1}';
    end
    % now perform gradient step
    eta = min(.1,100/i);
    for l = 1:nlayers
        W{l}=W{l}-eta*dW{l};
    end
    if mod(i,100)==0
        norm(Y-sign(backprop_predict(X,W)),1)
    end
end

function z = lossgrad(v,y)
%z = 2*(v-y); %squared loss: l = (v-y)^2
if v*y>1 %hinge loss: l = max(0,1-v*y)
    z = 0;
else
    z = -y;
end


function z=nonlin(x)
z = max(x,0);

function z=nonlingrad(x)
z = (sign(x)+1)/2;

function y = backprop_predict(X,W)
% forward propagation
nlayers = length(W);
n = size(X,1);
y = zeros(n,1);
v={};
for i = 1:n
    v{nlayers+1}=X(i,:)';
    for l = (nlayers):-1:2
        v{l}=nonlin(W{l}*v{l+1});
    end
    y(i)=W{1}*v{2};
end


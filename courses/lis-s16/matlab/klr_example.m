% This script illustrates kernelized linear regression
% Author: Andreas Krause (ETHZ)
% Note that the hyperparameters are manually chosen here;
% In practice, one would need to use cross-validation to fit them

n = 100; 
x = linspace(0,10,n);  %create 1-D input data
f = x+sin(x*5)-cos(x*10); %true functional response
y = f+0.2*randn(1,n); %generate noisy labels
clf
hold on
axis([0,10,-2,12])

% select a subset of points for training and plot
trainind = 1:50;
ntr = length(trainind);

xtr = x(trainind);
ytr = y(trainind);
plot(xtr,ytr,'ro');
title('Training data','fontsize',16)
pause

plot(x,f);
title('Function we wish to learn','fontsize',16)
pause

% we now use linear (ridge) regression for predictions
Ktr = xtr'*xtr; %construct kernel matrix on training data
lambda = .05;
alpha = (Ktr+lambda*eye(ntr))\ytr'; %solve for dual vars
Kpr = xtr'*x; %compute kernel between train and test points
ypr = alpha'*Kpr; %predict
title('Linear kernel','fontsize',16)
h = plot(x,ypr,'k','linewidth',2);

pause

delete(h)

% we now use KLR with Gaussian kernel for predictions
Dtr = dist(xtr);
Ktr = exp(-Dtr.^2*25);
lambda = .05;
alpha = (Ktr+lambda*eye(ntr))\ytr';
Dpr = dist(xtr',x);
Kpr = exp(-Dpr.^2*25);
ypr = alpha'*Kpr;
title('Gaussian kernel','fontsize',16)
h = plot(x,ypr,'k','linewidth',2);

pause

% we now use KLR with Gaussian + linear kernel for predictions
delete(h)
Dtr = dist(xtr);
Ktr = exp(-Dtr.^2*25)+xtr'*xtr;
lambda = .05;
alpha = (Ktr+lambda*eye(ntr))\ytr';
Dpr = dist(xtr',x);
Kpr = exp(-Dpr.^2*25)+xtr'*x;
ypr = alpha'*Kpr;
title('Gaussian + linear kernel','fontsize',16)
h = plot(x,ypr,'k','linewidth',2);

pause

% we now use KLR with Periodic kernel for predictions
delete(h)
Dtr = dist(xtr);
Ktr = exp(-sin(Dtr*2.5).^2);
lambda = .1;
alpha = (Ktr+lambda*eye(ntr))\ytr';
Dpr = dist(xtr',x);
Kpr = exp(-sin(Dpr*2.5).^2);
ypr = alpha'*Kpr;
title('Periodic kernel','fontsize',16)
h = plot(x,ypr,'k','linewidth',2);

pause

% we now use KLR with Periodic + linear kernel for predictions
delete(h)
Dtr = dist(xtr);
Ktr = exp(-sin(Dtr*2.5).^2)+xtr'*xtr;
lambda = .1;
alpha = (Ktr+lambda*eye(ntr))\ytr';
Dpr = dist(xtr',x);
Kpr = exp(-sin(Dpr*2.5).^2)+xtr'*x;
ypr = alpha'*Kpr;
title('Periodic + linear kernel','fontsize',16)
h = plot(x,ypr,'k','linewidth',2);

hold off

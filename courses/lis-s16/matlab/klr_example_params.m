% This script illustrates kernelized linear regression
% Author: Andreas Krause (ETHZ)
% Note that the hyperparameters are manually chosen here;
% In practice, one would need to use cross-validation to fit them

fctsmall = 100000;
fctlarge = 10;

n = 100; 
x = linspace(0,10,n);  %create 1-D input data
f = x+sin(x*5)-cos(x*10); %true functional response
y = f+0.2*randn(1,n); %generate noisy labels
clf
hold on
axis([0,10,-2,12])

% select a subset of points for training and plot
trainind = 1:2:50;
ntr = length(trainind);

xtr = x(trainind);
ytr = y(trainind);
plot(xtr,ytr,'ro');
title('Training data','fontsize',16)
pause

plot(x,f);
title('Function we wish to learn','fontsize',16)
pause

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
delete(h)

% now use too small bandwidth
Dtr = dist(xtr);
Ktr = exp(-Dtr.^2*fctsmall);
lambda = .05;
alpha = (Ktr+lambda*eye(ntr))\ytr';
Dpr = dist(xtr',x);
Kpr = exp(-Dpr.^2*fctsmall);
ypr = alpha'*Kpr;
title('Gaussian kernel, small h','fontsize',16)
h = plot(x,ypr,'k','linewidth',2);

pause 
delete(h)
% now use too large bandwidth
Dtr = dist(xtr);
Ktr = exp(-Dtr.^2/fctlarge);
lambda = .05;
alpha = (Ktr+lambda*eye(ntr))\ytr';
Dpr = dist(xtr',x);
Kpr = exp(-Dpr.^2/fctlarge);
ypr = alpha'*Kpr;
title('Gaussian kernel, large h','fontsize',16)
h = plot(x,ypr,'k','linewidth',2);

pause

hold off

% This script illustrates gradient descent for linear regression
% Author: Andreas Krause (ETHZ)

a = 1; %slope
eps = .2; %noise
X = randn(100,1); %randomly generate features X
Y = a*X+randn(100,1)*eps; %generate noisy response Y
w = 0; %initial guess for slope
for i = 1:20
    % some plotting
    plot(X,Y,'.');
    hold on
    Xint = linspace(min(X),max(X));
    Yint=w*Xint;
    axis([-2,2,-2,2]);
    plot(Xint,Yint,'r');
    hold off
    pause
   
    eta=(1e-2)/i; %set learning rate
    w=w+eta*2*(Y-w*X)'*X; %gradient descent
    
end
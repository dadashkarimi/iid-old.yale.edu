%Demo for linear regression via gradient descent
%for fitting polynomials
% Author: Andreas Krause (ETHZ)

degree = 3; %degree of polynomial
n=100; %number of data points
minX = -3; maxX = 3; %for plotting
a = [-1; 1;.5;-.5]; %coefficients
eps = .2; %noise
xs = randn(n,1); %1D x-coordinate
X = ones(n,1); %feature vectors
lambda = 0; %ridge regression regularization parameter
for i = 1:degree
    X = [X xs.^i]; %creating polynomial features
end
Y = X*a+randn(100,1)*eps; %generate noisy response Y
n_iter = 200; %number of iterations
d = size(X,2); %number of dimensions
w = zeros(d,1); %initial guess for slope
eta0 = 1e-2; %Initial step size for gradient descent
TOL = 1e-2; %tolerance for termination
wold = inf*zeros(d,1);
fval = inf;
for i = 1:n_iter
    % some plotting
    plot(X(:,2),Y,'.');
    hold on
    Yint=X*w;
    [~,I]=sort(X(:,2));
    axis([minX,maxX,minX,maxX]);
    plot(X(I,2),Yint(I),'r');
    hold off
    pause(.1)
    fvalold = fval; 
    fval = norm(Y-X*w);
    if fval<fvalold %bold driver for updating step size
        eta = eta*1.1
    else
        eta = eta/2
    end
    wold = w; %save old weights for termination check
    grads = 2*(Y-X*w)*ones(1,d).*X; %compute gradients of LS for each data
    w=w+eta*(sum(grads)'-2*lambda*w); %gradient descent
    wdiff = norm(wold-w);
    if wdiff<TOL %termination condition
        break
    end
end
disp(sprintf('finished after %d iterations',i))

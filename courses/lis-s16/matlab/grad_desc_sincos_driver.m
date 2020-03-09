% Demo of linear regression by gradient descent
% Stepsize chosen via "bold driver" heuristic
% Author: Andreas Krause (ETHZ)

clf
minX = -3; maxX = 3; %for plotting
n_iter = 20; %number of iterations
eta0 = .1; %initial learning rate
addpath(fullfile(docroot,'techdoc','creating_plots','examples'))
fval = 10;
eta=eta0; 

for i = 1:n_iter
    % some plotting
    [Xg,Yg]=meshgrid(minX:.1:maxX,minX:.1:maxX);
    wg = [Xg(:),Yg(:)];
    for j=1:size(wg,1)
        wj = wg(j,:)';
        Zg(j)=sin(wj(1))*cos(wj(2));
    end
    Zg = reshape(Zg,size(Xg,1),size(Xg,2));
    contour(Xg,Yg,Zg,50);
    hold on
    
    if i==1 %collect starting point
        w = ginput(1)'
    end
    fvalold = fval;
    fval = sin(w(1))*cos(w(2));
    if fval>fvalold %"bold driver" update
        eta=eta/5
    else
        eta=eta*1.1
    end
       
    grad = [cos(w(1))*cos(w(2));-sin(w(2))*sin(w(1))];
    wold = w;
    w=w-eta*grad; %gradient descent
    [figX,figY] = dsxy2figxy(wold,w);
    harr = annotation('arrow',[figX(1),figY(1)],[figX(2),figY(2)]);
    hold off
    pause
    delete(harr)
end
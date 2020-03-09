% PCA Demo
% Andreas Krause (ETHZ)

clear;
n = 400;
data = 1;

if data == 1
    use_std=[.5 .5];
%    use_std=[10 .1]; % might try this instead
    npc = n/2;

    X = [randn(npc,2)*diag(use_std)+ones(npc,2); randn(npc,2)*diag(use_std)-ones(npc,2)];
    X = X/std(X(:));
elseif data == 2
    use_std=.1;
    npc = n/2;
    X(1:npc,:)=randn(npc,2);
    X(1:npc,:)=X(1:npc,:)./(sqrt(sum(X(1:npc,:).^2,2))*ones(1,2));
    X(1:npc,:)=X(1:npc,:)+randn(npc,2)*use_std;
    Y(1:npc)=1;
    X((npc+1):n,:)=randn(npc,2)*use_std;
end


%let's center the data:
X = X-ones(n,1)*mean(X);

% %now do PCA
[U,S,V]=svd(X); %obtain singular value decomposition
% S is an nxn matrix of singular values; 
% since X is two-dimensional, all but the top two singular values are 0
S(2,2)=0; %zero-out the second (smaller) non-zero singular value
Xp = U*S*V'; %reconstruct X
                  
scatter(X(:,1),X(:,2)); hold on
scatter(Xp(:,1),Xp(:,2)); hold off

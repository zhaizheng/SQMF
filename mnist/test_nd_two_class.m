clear
addpath('../')
data = load('mnist.mat'); 
Y  = data.trainY;
IND = [];
classes = [4,9];
C = [];
for i = 1:2
    ttt = find(Y==classes(i));
    IND = [IND, ttt(1:150)];
    C = [C,i*ones(1,150)];
end


%%
X = double(data.trainX)'; 
X = X(:,IND);



q = mean(X,2);
neig = 295;
d = 3;

M = 100;


Xm = X-mean(X,2);
[U, ~, ~] = svd(Xm);
Rc = U(:,1:d)*U(:,1:d)'*Xm+mean(X,2);


rho = 0; k = neig; n = neig; delta = 0;

%[Tau_p, Data_p, ~, ~] = initial_Tau(q, d, X, k, n);  
Tau = X-mean(X,2);
[~, ~, V] = svd(Tau);
Tau = qrs(V(:,1:d)');
%[f,~] = fit_nonlinear(X, Tau, rho, delta, 0);  
%[Q_F, x0_F, Theta_F, Tau_F, error] = Factorization(X,d);
ND = [1,2,3,4,5,6];
INDS  = sort(randperm(300,M));
Final = [];
%%
for t = 1:length(ND)
    total  = 100; alg = 3; lambda = 0.8; nd = ND(t);
    [Q_F, x0_F, Theta_F, Tau_F, error] = Factorization3(X,d,total,alg,lambda, nd);

    RREE = zeros(3,length(ND));
    for i = 1:length(INDS)
        k = INDS(i);
        RREE(1,i) = norm(X(:,k)-Rc(:,k))^2;
        [~, M_F] = Psi(Tau_F(:,k), Theta_F);
        mg_vec  = x0_F + Q_F*M_F;
       % mg_vec = f.Parm*Construct_Higher_Order(embedding(:,k));
        RREE(2,i) = norm(X(:,k)-mg_vec)^2; 
        RREE(3,i) = (RREE(1,i)-RREE(2,i))/RREE(1,i);
    end
    Final = [Final,[mean(RREE(1,:)),mean(RREE(2,:)),mean(RREE(3,:))]'];
end


function Tau = qrs(Tau)
    d = size(Tau, 1);
    [Q,~] = qr([ones(size(Tau, 2),1),Tau']);
    %Tau = size(Tau,2)*Q(:,2:d+1)';
    Tau = Q(:,2:d+1)';
end



function [psi, M] = Psi(Phi, Theta)
    d = size(Phi,1);
    n = size(Phi,2);
    psi = [];%zeros(d*(d+1)/2,n);]
    for i = 1:d
        for j = i:d
            psi = [psi; Phi(i,:).*Phi(j,:)];
        end
    end
    M = [Phi; Theta'*psi];
end
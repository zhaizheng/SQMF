%%
addpath('../')
data = load('mnist.mat');
X = double(data.trainX)';
imshow(uint8(reshape(X(:,1),[28,28]))');
%%
Y = data.trainY;
NX = X+20*randn(size(X));
IND = find(Y==5);
ind = IND(1);
neig = 300;
d = 2;
Y = find_nearest(X(:,ind),X, neig);
total = 100; alg = 3; lambda = 0.3; nd = 2;
[Q, x0, Theta, Tau, error] = Factorization3(Y, d, total, alg, lambda, nd);
%%
ti = (Q(:,1:d))'*(X(:,ind)-x0);
[~, M] = Psi(ti, Theta);
IMG  = x0 + Q*M;
%%
figure(1)
subplot(1,3,1)
image(uint8(reshape(IMG,[28,28]))');
subplot(1,3,2)
NX = uint8(X+10*randn(size(X)));
image(uint8(reshape(NX(:,ind),[28,28]))');
subplot(1,3,3)
image(uint8(reshape(X(:,ind),[28,28]))');

%%
V = [0,1;1,0;1,1;1,-1];
K = -1:0.07:1;
K2 = -1:.2:1;
figure(2)
for t = 1:4
    if t == 1
        K = K2;
    end
    for i = 1:length(K)
        temp_coord = 500*V(t,:)'*(K(i)-0.5);
        [~, M] = Psi(temp_coord, Theta);
        Moving_IMG  = x0 + Q*M;
        axes('position',[(V(t,1)*K(i)+0.9)/2,(V(t,2)*K(i)+0.9)/2,.1,.1]);
        image(uint8(reshape(Moving_IMG,[28,28]))')
        axis off
        hold on
    end
end





function Y = find_nearest(x, X, k)
    d = sum((X-x).^2,1);
    [~,ind] = sort(d,'ascend');
    Y = X(:,ind(1:k));
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
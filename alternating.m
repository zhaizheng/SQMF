
c = -2:0.1:2;
y = c.^2;
X = [c;y]+rand(2,length(c));
plot(X(1,:),X(2,:),'*');




function alternating(X)
    d = 1;
    D = size(X,1);
    Theta = zeros(d*(d+1)/2, D-d);
    Q = eye(D);
    x0 = mean(X,2);
    while true
        Phi = Projection(x0, Q, Theta, X, d);
        [Q, x0, Theta] = Regression(Phi, X);
        [~, M] = Psi(Phi, Theta);
        re = norm(X-c-Q*M)
    end
end




function Tau = Projection(x0, Q, Theta, X, d) 
    n = size(X,2);
    Tau = zeros(d,n);
    D = size(X,1);
    U = Q(:,1:d);
    Up = Q(:,d+1:D);
    A = toTensor(Theta, d);
    for i = 1:n
        s = U'*(X(:,i)-x0);
        c = Up'*(X(:,i)-x0);
        tau = zeros(d,1);
        step = 1;
        while ture 
            % while f_value(tau-(step*gradient(tau, s, c, A)), s, c, A) > f_value(tau,s,c,A)
            %     step = step/2;
            % end
            [g,H ] = gradient(tau, s, c, A);
            %tau = tau - step*gradient(tau, s, c, A);
            tau = tau - H\g;
        end
        Tau(:,i) = tau;
    end
end

function [g,H] = gradient(tau, s, c, A)
    dim = size(A,1);
    d = length(tau);
    r = zeros(dim,1);
    for i = 1:dim
        r(i) = tau'*sequeeze(A(i,:,:))*tau;
    end
    Mr = zeros(dim,d);
    for i = 1:dim
        Mr(dim,:)  = sequeeze(A(dim,:,:))*tau;
    end
    M2 = zeros(d,d);
    for i = 1:dim
        M2 = M2 + sequeeze(A(i,:,:))*(r(i)-c(i));
    end
    g = 2*(tau-s)+4*Mr'*(r-c);
    H = 2*eye(d)+8*Mr'*Mr+4*M2;
end

function f = f_value(tau, s, c, A)
    dim = size(A,1);
    r = zeros(dim,1);
    for i = 1:dim
        r(i) = tau'*sequeeze(A(i,:,:))*tau;
    end
    f = (s-tau)'*(s-tau)+(c-r)'*(c-r);
end


function A = toTensor(Theta, d)
    dh = size(Theta,2);
    A = zeros(dh,d,d);
    for i = 1:dh
        A(i,:,:) = vector2mat(Theta(:,i),d);
    end
end

function M = vector2mat(a, dim)
    M = zeros(dim);
    W = ones(dim);
    ind = triu(W)>0;
    M(ind) = a/2;
    S = M+M';
end





function [Q, c, Theta] = Regression(Phi, X)
    d = size(Phi,1);
    D = size(X,1);
    Theta = zeros(d*(d+1)/2, D-d);
    Q = eye(D);
    while true
        [psi, M] = Psi(Phi, Theta);
        c = mean(X-Q*M,2);
        [U1, ~, U2] = svd((X-c)*M');
        Q_new = U1*U2';
        V = Q_new(:,d+1:D);
        Theta_new = (psi*psi')\psi*(X-c)'*V;
        error = norm(Theta_new-Theta)+norm(Q_new-Q);
        if error < 1.e-7
            break;
        end
        Q = Q_new;
        Theta = Theta_new;
    end
end


function [psi, M] = Psi(Phi, Theta)
    d = size(Phi,1);
    n = size(Phi,2);
    psi = zeros(d*(d+1)/2,n);
    for i = 1:d
        for j = i:d
            psi((i-1)*d+j,:) = Phi(i,:).*Phi(j,:);
        end
    end
    M = [Phi; Theta'*psi];
end
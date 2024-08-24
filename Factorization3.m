function [Q, c, Theta, Tau, error,error2] = Factorization3(X,d,total,alg,lambda, nd)
    if nargin<3
        total = 100; alg = 1; lambda = 0; 
    end
    D = size(X,1);
    CX = X-mean(X,2);
    [U,~,~] = svd(CX);
    Tau = U(:,1:d)'*CX;
    n = 1; 
    Dd = min((d^2+d)/2,D-d);
    Dd = min(Dd, nd);
    Theta = rand(d*(d+1)/2, Dd);
    % Q = eye(D);
    Q = U(:,1:d+Dd);
    error2 = [];
    % 
    % [psi, ~] = Psi(Tau, Theta);
    % c = mean(X,2);
    % V = U(:,d+1:d+Dd);
    % Theta= 1/(lambda+1)*pinv(psi*psi')*psi*(X-c)'*V;
    while true
        [Q, c, Theta, error] = Regression(Tau, X, Theta, Q, lambda, nd);
        Tau_new = Projection(c, Q, Theta, X, d, Tau, alg, nd, lambda);

        [psi, M] = Psi(Tau_new, Theta);
        error2 = [error2, norm(X-c-Q*M,'fro')^2+lambda*norm(Theta'*psi,'fro')^2];
        % temp = qr(Tau_new');
        % Tau_new = temp(:,1:d)';
        %fprintf('Tau change:%3.5f\n',norm(Tau_new-Tau,'fro'));
        if norm(Tau_new-Tau,'fro')<1.e-7*norm(Tau_new,'fro') || n > total
            break;
        end
        Tau = Tau_new;
        n = n+1;
        fprintf('iter=%d,error:%3.4f\n',n, error2(end));
    end
end



function Tau = Projection(x0, Q, Theta, X, d, Tau, alg, nd, lambda) 
    n = size(X,2);
    D = size(X,1);
    U = Q(:,1:d);
    Dd = min((d^2+d)/2,D-d);
    Dd = min(Dd, nd);
    Up = Q(:,d+1:d+Dd);
    A = toTensor(Theta, d);
    for i = 1:n
        s = U'*(X(:,i)-x0);
        c = Up'*(X(:,i)-x0);
        tau = zeros(d,1);%Tau(:,i);
        if alg == 1
            for iter = 1:1000
                [g, H] = gradient(tau, s, c, A, lambda);
                tau = tau - pinv(H)*g;   
                if norm(g)<1.e-5
                    break;
                end
                %fprintf('projection tau %d,step: %f,norm of gradient:%f\n', i, step, norm(gradient(tau, s, c, A)));
            end
        elseif alg == 2
            for iter = 1:40
                [g, ~] = gradient(tau, s, c, A, lambda);
                step = 1;
                while  f_value(tau-step*g, s, c, A, lambda) > f_value(tau, s, c, A, lambda)
                    step = step/2;
                end
                tau = tau-step*g;
            end
        else
        %    tau = zeros(size(s));
            for iter = 1:1000 
                tau_new =  surrogate(tau, s, c, A, lambda);
                if norm(tau_new-tau)<1.e-8
                    break
                end
                tau = tau_new;
            end
        end
        Tau(:,i) = tau;
        %fprintf('projection tau %d,norm of gradient%f\n', i, norm(gradient(tau, s, c, A)));
    end
end

function tau = surrogate(tau, s, c, A, lambda)
    dim = size(A,1);
    d = length(tau);
    r = zeros(dim,1);
    Mr = zeros(dim,d);
    for i = 1:dim
        Mr(i,:)  = squeeze(A(i,:,:))*tau;
    end
    B = zeros(d,d);
    for j = 1:dim
        B = B + c(j)*squeeze(A(j,:,:));
    end
    tau = inv(2*(1+lambda)*Mr'*Mr+eye(d))*(2*B*tau+s);
end


function [g,H] = gradient(tau, s, c, A, lambda)
    dim = size(A,1);
    d = length(tau);
    r = zeros(dim,1);
    for i = 1:dim
        r(i) = (1+lambda)*tau'*squeeze(A(i,:,:))*tau;
    end
    Mr = zeros(dim,d);
    for i = 1:dim
        Mr(i,:)  = squeeze(A(i,:,:))*tau;
    end
    M2 = zeros(d,d);
    for i = 1:dim
        M2 = M2 + squeeze(A(i,:,:))*(r(i)-c(i));
    end
    g = 2*(tau-s)+4*Mr'*(r-c);
    H = 2*eye(d)+8*(1+lambda)*Mr'*Mr+4*M2;
end


function f = f_value(tau, s, c, A, lambda)
    dim = size(A,1);
    r = zeros(dim,1);
    for i = 1:dim
        r(i) = tau'*squeeze(A(i,:,:))*tau;
    end
    f = (s-tau)'*(s-tau)+(c-r)'*(c-r)+lambda*r'*r;
end


function A = toTensor(Theta, d)
    dh = size(Theta,2);
    A = zeros(dh,d,d);
    for i = 1:dh
        A(i,:,:) = vector2mat(Theta(:,i),d);
    end
end


function S = vector2mat(a, dim)
    M = zeros(dim);
    W = ones(dim);
    ind = triu(W)>0;
    M(ind) = a/2;
    S = M+M';
end


function [Q, c, Theta, error] = Regression(Phi, X, Theta, Q, lambda, nd)
    d = size(Phi,1);
    D = size(X,1);
    Dd = min((d^2+d)/2,D-d);
    Dd = min(Dd, nd);
    error = [];
    while true
        [psi, M] = Psi(Phi, Theta);
        c = mean(X-Q(:,1:d+Dd)*M,2);
        [U1,~,V1] = svd((X-c)*M');
        Q_new = U1(:,1:d+Dd)*V1';%U1(:,1:d+Dd)*V1(:,1:d+Dd)';
        %Q_new = Temp(:,1:d+Dd);
        V = Q_new(:,d+1:d+Dd);
        %Theta_new = inv(psi*psi'+eye((d^2+d)/2))*psi*(X-c)'*V;
        %[~,D,~] = svd(psi*psi');
        %sm = min(diag(D));
        Theta_new = 1/(lambda+1)*pinv(psi*psi')*psi*(X-c)'*V;
        error1 = norm(Theta_new-Theta,'fro');%+norm(Q_new-Q);
        %fprintf('f_value%f,error=%f\n',norm(X-c-Q*M),error)
        if error1 < 1.e-10
            break;
        end
        Q = Q_new;
        Theta = Theta_new;
        [psi, M] = Psi(Phi, Theta);
        error = [error,norm(X-c-Q*M,'fro')^2+lambda*norm(Theta'*psi,'fro')^2];
        %fprintf('regression error:%3.5f\n',error);
    end
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


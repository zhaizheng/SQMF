function Tau = Projection(x0, Q, Theta, X, d, Tau, Newton) 
    n = size(X,2);
    D = size(X,1);
    U = Q(:,1:d);
    Up = Q(:,d+1:D);
    A = toTensor(Theta, d);
    for i = 1:n
        s = U'*(X(:,i)-x0);
        c = Up'*(X(:,i)-x0);
        tau = s;
        %tau = Tau(:,i);
        step = 5;
        if Newton == 1
            for iter = 1:100
                [g, H] = gradient(tau, s, c, A);
                tau = tau - pinv(H)*g;   
                if norm(g)<1.e-5
                    break;
                end
                %fprintf('projection tau %d,step: %f,norm of gradient:%f\n', i, step, norm(gradient(tau, s, c, A)));
            end
        else
            for iter = 1:100 
                tau_new =  surrogate(tau, s, c, A);
                if norm(tau_new-tau)<1.e-5
                    break
                end
                tau = tau_new;
            end
        end
        Tau(:,i) = tau;
        %fprintf('projection tau %d,norm of gradient%f\n', i, norm(gradient(tau, s, c, A)));
    end
end

function tau = surrogate(tau, s, c, A)
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
    tau = pinv(2*Mr'*Mr+eye(d))*(B*tau+s);
end


function [g,H] = gradient(tau, s, c, A)
    dim = size(A,1);
    d = length(tau);
    r = zeros(dim,1);
    for i = 1:dim
        r(i) = tau'*squeeze(A(i,:,:))*tau;
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
    H = 2*eye(d)+8*Mr'*Mr+4*M2;
end


function f = f_value(tau, s, c, A)
    dim = size(A,1);
    r = zeros(dim,1);
    for i = 1:dim
        r(i) = tau'*squeeze(A(i,:,:))*tau;
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


function S = vector2mat(a, dim)
    M = zeros(dim);
    W = ones(dim);
    ind = triu(W)>0;
    M(ind) = a/2;
    S = M+M';
end
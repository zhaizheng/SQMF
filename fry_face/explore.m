addpath('./../')
addpath('./../methods/')
data = load('frey_rawface.mat');
X = double(data.ff);


ind = 600;
neig = 300;
d = 3;

Sigma = 30;
K = 32:4:60;
error = zeros(length(Sigma),length(K),6,5);
Ind_rand = 31:40;%randperm(1000);
Ind = Ind_rand(1:10);
NX = X+Sigma*rand(size(X));


%%
figure(1)
tl = tiledlayout(4,5,'TileSpacing','compact');
% for n = 1:length(Ind)
%     nexttile
%     imshow(uint8(reshape(X(:,Ind(n)),[20,28]))');
% end



% for n = 1:length(Ind)
%     nexttile
%     imshow(uint8(reshape(NX(:,Ind(n)),[20,28]))');
% end
%figure(2)
%t2 = tiledlayout(1,6,'TileSpacing','compact');

R5 = cell(length(Sigma),length(K),length(Ind));
for s = 1:length(Sigma)
    for k = 2 %length(K)
        for n = 1:length(Ind)
            ind = Ind(n);
            input = NX(:,ind);
            in_true = X(:,ind);
            lambda = 1; alg = 3; nd = 3;
            [R1,error, error2] = QMF(input, NX, K(k), d, nd, lambda, alg);
            nexttile
            plot(1:length(error), error); title('inner')
            nexttile
            plot(1:length(error2),error2); title('outer')
            [R5{s,k,n},~] = PCA_refine(input, NX, K(k), d);
            fprintf('Sigma = %f, K = %d, sample ind=%d, fitting error=%f,PCA error=%f, original error=%f\n',...
                Sigma(s),K(k),n,norm(R1-in_true),norm(R5{s,k,n}-in_true) ,norm(input-in_true));
            %      fprintf('Sigma = %f, K = %d, sample ind=%d, fitting error=%f PCA error=%f, original error=%f\n',...
            %    Sigma(s),K(k),n,min(sum((X-R1).^2,1)),min(sum((X-R5{s,k,n}).^2,1)),min(sum((X-input).^2,1)));
            % nexttile
            % imshow(uint8(reshape(R1,[20,28]))');
        end
    end
end

% for n = 1:length(Ind)
%     nexttile
%     imshow(uint8(reshape(R5{s,k,n},[20,28]))');
% end



 
function [R, error, error2] = QMF(input, X, neig, d, nd, lambda, alg)
   
    Y = find_nearest(input, X, neig);
    [Q, x0, Theta, ~, error, error2] = Factorization3(Y, d, 60, alg, lambda, nd);
    %figure; plot(error)
    ti = (Q(:,1:d))'*(input-x0);
    %ti = zeros(d,1);
    ti_new = Projection(x0, Q, Theta, input, d, ti, alg, nd);
    [~, M] = Psi(ti, Theta);
    %[~, M] = Psi(ti_new, Theta);
    R = x0 + Q*M;
end


function Tau = Projection(x0, Q, Theta, X, d, Tau, alg, nd) 
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
        tau = s;
        %tau = Tau(:,i);
        step = 5;
        if alg == 1
            for iter = 1:100
                [g, H] = gradient(tau, s, c, A);
                tau = tau - pinv(H)*g;   
                if norm(g)<1.e-5
                    break;
                end
                %fprintf('projection tau %d,step: %f,norm of gradient:%f\n', i, step, norm(gradient(tau, s, c, A)));
            end
        elseif alg == 2
            for iter = 1:100
                [g, ~] = gradient(tau, s, c, A);
                while  f_value(tau-step*g, s, c, A) > f_value(tau, s, c, A)
                    step = step/2;
                end
                tau = tau-step*g;
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


function f = f_value(tau, s, c, A)
    dim = size(A,1);
    r = zeros(dim,1);
    for i = 1:dim
        r(i) = tau'*squeeze(A(i,:,:))*tau;
    end
    f = (s-tau)'*(s-tau)+(c-r)'*(c-r);
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


function T = Tangent_Estimation(tau, U, V, Theta, d, D)

    A = toTensor(Theta, d);
    R = zeros(D-d,d);
    for k = 1:d
        R = R+A(:,:,k)*tau(k);
    end
    R = U+2*V*R;
    [Td,~] = qr(R);
    T = Td(:,1:d);

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
end

function [e,et] = Compute_Error(Z, ZT, T)
    e = norm(Z-T,'fro')^2;
    D = size(T,1);
    et = 0;
    for i = 1:size(T,2)
        et = et + norm(eye(D)-T(:,i)*T(:,i)'/(norm(T(:,i))^2)-ZT{i},'fro')^2;
    end
end



function [data_true, data] = generate_data3(sigma, num)
    rdata = randn(3,num);
    E = diag(1./sqrt(sum(rdata.^2,1)));
    data_true = rdata*E;
    data = data_true+sigma*randn(3,num);
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
%%
addpath('./../')
addpath('./../methods/')
data = load('frey_rawface.mat');
X = double(data.ff);


ind = 600;
neig = 300;
d = 2;

Sigma = 30;
K = 16;%:4:32;
error = zeros(length(Sigma),length(K),6,5);
Ind = 38:40;%31:40;%[100,200,300,400,500,600];


for s = 1 %1:length(Sigma)
    NX = X+Sigma(s)*rand(size(X));
    for k = 1:length(K)
        figure(k)
        tl = tiledlayout(length(Ind),7,'TileSpacing','compact');
        for n = 1:length(Ind)
            ind = Ind(n);
            input = NX(:,ind);
            in_true = X(:,ind);
            R1 = QMF(input, NX, K(k), d);
            [R2,~,~] = MovingLS(NX, input, K(k), d);
            %%
            [R3,~] = linear_log_KDE(K(k), NX, input, d);
            [R4,~] = linear_mfit(NX, input, d, K(k));
            [R5,~] = PCA_refine(input, NX, K(k), d);
            error(s,k,n,1) = norm(R1-in_true,'fro')^2;
            error(s,k,n,2) = norm(R2-in_true,'fro')^2;
            error(s,k,n,3) = norm(R3-in_true,'fro')^2;
            error(s,k,n,4) = norm(R4-in_true,'fro')^2;
            error(s,k,n,5) = norm(R5-in_true,'fro')^2;
            fprintf('error1:%f,error2:%f,error3:%f,error4:%f,error5:%f\n', ...
                norm(R1-in_true).^2,norm(R2-in_true).^2,norm(R3-in_true).^2,norm(R4-in_true).^2,norm(R5-in_true).^2)
            nexttile
            imshow(uint8(reshape(input,[20,28]))');
            nexttile
            imshow(uint8(reshape(in_true,[20,28]))');
            nexttile
            imshow(uint8(reshape(R1,[20,28]))');
            nexttile
            imshow(uint8(reshape(R2,[20,28]))');
            nexttile
            imshow(uint8(reshape(R3,[20,28]))');
            nexttile
            imshow(uint8(reshape(R4,[20,28]))');
            nexttile
            imshow(uint8(reshape(R5,[20,28]))');
        end
    end
end

Final_result = [];
for k = 1:length(K)
    Final_result = [Final_result; mean(squeeze(error(1,k,:,:)),1)];
end

%%





















%%

% K = 14:4:50;
% %K = 30;
% Sigma = 0.08:0.02:0.12;
% d = 2;
% D = 3;
% num = 150;
% e11 = zeros(length(Sigma),length(K));
% e12 =  zeros(length(Sigma),length(K));
% e21 = zeros(length(Sigma),length(K));
% e22 =  zeros(length(Sigma),length(K));
% 
% Z = cell(5,length(Sigma),length(K));
% T = cell(1,length(Sigma));
% for s = 1:length(Sigma)
%     sigma = Sigma(s);
%     [T{s}, X] = generate_data3(sigma, num);
%     for k = 1:length(K)
%         neig = K(k);
%         [Z{1,s,k},ZT{1,s,k}] = QMF(X, neig, d);
%         [Z{2,s,k},ZT{2,s,k},~] = MovingLS(X, X, neig, d);
%         [Z{3,s,k},ZT{3,s,k}] = linear_log_KDE(neig,X,X,D,d);
%         [Z{4,s,k},ZT{4,s,k}] = linear_mfit(X,X,d,neig);
%         [Z{5,s,k},ZT{5,s,k}] = PCA_refine(X,X,neig,d);
%     end
% end
% %%
% Methods = {'QMF','MLS','LKDE','MFIT','PCA'};
% for t = 1:5
%     for s = 1:length(Sigma)
%         for k = 1:length(K)
%            [e(t,s,k),et(t,s,k)] = Compute_Error(Z{t,s,k}, ZT{t,s,k}, T{s});   
%         end
%     end
%     fprintf('Approximation:%s\n',Methods{t});
%     format_print(reshape(e(t,:,:),length(Sigma),length(K))/num)
%     fprintf('T Approximation:%s\n',Methods{t});
%     format_print(reshape(et(t,:,:),length(Sigma),length(K))/num)
% end
% %%
% RE = []
% RET = []
% for t = 1:5
%     RE = [RE;reshape(e(t,:,:),length(Sigma),length(K))/num];
%     RET = [RET;reshape(et(t,:,:),length(Sigma),length(K))/num];
% end
% format_print(RE)
% fprintf('\n')
% format_print(RET)
% 
% format_print([RE,RET])
%%
%format_print([RE(:,4:end),RET(:,4:end)])

% function [Z,ZT] = QMF(X, neig, d)
%        Z = zeros(size(X));
%        ZT = cell(1,size(X,2));
%        D = size(X,1);
%        for i = 1:size(X,2)
%             Y = find_nearest(X(:,i),X, neig);
%             [Q, x0, Theta, ~, ~] = Factorization(Y,d);
%             ti = (Q(:,1:d))'*(X(:,i)-x0);
%             [~, M] = Psi(ti, Theta);
%             Z(:,i)  = x0 + Q*M;
%             T = Tangent_Estimation(ti, Q(:,1:d), Q(:,d+1:D), Theta, d, D);
%             ZT{i} = T*T';%Q(:,1:d)*Q(:,1:d)';
%             %ZT{i} = T*T';
%             %Pr = eye(D)-X(:,i)*X(:,i)'/(norm(X(:,i))^2);
%             %re_tangent(k,s) = re_tangent(k,s)+norm(P-Pr,'fro')^2;
%             fprintf('the %d th sample\n',i);
%        end
%         %re(k,s) = norm(Z-T,'fro')^2;
%         %fprintf('K=%d,s=%d,error=%.3f,error2=%.3f\n',k,s,re(k,s),re_tangent(k,s));
% end


function R = QMF(input, X, neig, d)
    alg = 3;
    nd = 3;
    lambda = 1;
    Y = find_nearest(input, X, neig);
    [Q, x0, Theta, ~, ~] = Factorization3(Y, d, 40, alg,lambda, nd);
    ti = (Q(:,1:d))'*(input-x0);
    ti_new = Projection(x0, Q, Theta, input, d, ti, alg, nd);
    [~, M] = Psi(ti_new, Theta);
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
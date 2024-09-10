addpath('./methods/')
addpath('./tools/')
K = 22:4:46;%;14:4:50;
%K = 30;
Sigma = 0.06:0.02:0.1;
d = 2;
D = 3;
num = 150;
e11 = zeros(length(Sigma),length(K));
e12 =  zeros(length(Sigma),length(K));
e21 = zeros(length(Sigma),length(K));
e22 =  zeros(length(Sigma),length(K));

Z = cell(6,length(Sigma),length(K));
T = cell(1,length(Sigma));
Time = zeros(1,6);
for s = 1:length(Sigma)
    sigma = Sigma(s);
    [T{s}, X] = generate_data3(sigma, num);
    for k = 1:length(K)
        neig = K(k);
        t1 = cputime;
        [Z{1,s,k},ZT{1,s,k}] = SQMF(X, neig, d);
        t2 = cputime;
        [Z{2,s,k},ZT{2,s,k},~] = MovingLS(X, X, neig, d);
        t3 = cputime;
        [Z{3,s,k},ZT{3,s,k}] = linear_log_KDE(neig,X,X,d);
        t4 = cputime;
        [Z{4,s,k},ZT{4,s,k}] = linear_mfit(X,X,d,neig);
        t5 = cputime;
        [Z{5,s,k},ZT{5,s,k}] = PCA_refine(X,X,neig,d);
        t6 = cputime;
        [Z{6,s,k},ZT{6,s,k}] = RQMF_(X, neig, d, 0.01);
        t7 = cputime;
        Time = Time + [t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6];
    end
end
Time = Time/length(Sigma)/length(K);
%%
e = zeros(6,length(Sigma),length(K));
et = zeros(6,length(Sigma),length(K));
Methods = {'SQMF','MLS','LKDE','MFIT','PCA','QMF'};
for t = 1:6
    for s = 1:length(Sigma)
        for k = 1:length(K)
           [e(t,s,k),et(t,s,k)] = Compute_Error(Z{t,s,k}, ZT{t,s,k}, T{s});   
        end
    end
    fprintf('Approximation:%s\n',Methods{t});
    format_print(reshape(e(t,:,:),length(Sigma),length(K))/num)
    fprintf('T Approximation:%s\n',Methods{t});
    format_print(reshape(et(t,:,:),length(Sigma),length(K))/num)
end
%%
RE = [];
RET = [];
for t = 1:6
    RE = [RE;reshape(e(t,:,:),length(Sigma),length(K))/num];
    RET = [RET;reshape(et(t,:,:),length(Sigma),length(K))/num];
end
format_print(RE)
fprintf('\n')
format_print(RET)

format_print([RE,RET])

fprintf('\n')
format_print([RE(:,1:end),RET(:,1:end)])

function [Z,ZT] = SQMF(X, neig, d)
       Z = zeros(size(X));
       ZT = cell(1,size(X,2));
       D = size(X,1);
       for i = 1:size(X,2)
            Y = find_nearest(X(:,i),X, neig);
            [Q, x0, Theta, ~, ~] = Factorization3(Y,d,40,3,0,1);
            ti = (Q(:,1:d))'*(X(:,i)-x0);
            ti = Projection(x0, Q, Theta, X(:,i), d, ti, 1);
            [~, M] = Psi(ti, Theta);
            Z(:,i)  = x0 + Q*M;
            T = Tangent_Estimation(ti, Q(:,1:d), Q(:,d+1:D), Theta, d, D);
            ZT{i} = T*T';%Q(:,1:d)*Q(:,1:d)';
            %ZT{i} = T*T';
            %Pr = eye(D)-X(:,i)*X(:,i)'/(norm(X(:,i))^2);
            %re_tangent(k,s) = re_tangent(k,s)+norm(P-Pr,'fro')^2;
            fprintf('the %d th sample\n',i);
       end
        %re(k,s) = norm(Z-T,'fro')^2;
        %fprintf('K=%d,s=%d,error=%.3f,error2=%.3f\n',k,s,re(k,s),re_tangent(k,s));
end

function [Z,ZT] = RQMF_(X, neig, d, rho)
       Z = zeros(size(X));
       ZT = cell(1,size(X,2));
       D = size(X,1);
       for i = 1:size(X,2)
            Y = find_nearest(X(:,i),X, neig);

            Temp = Y-mean(Y,2);
            [~, ~, V] = svd(Temp);
            Tau = qrs(V(:,1:d)');
            
            [f, ~] = RQMF(Y, Tau, rho, 0);
            tau = projection(X(:,i), f.A, f.B, f.c, zeros(d,1));
            Z(:,i) = f.Parm*Construct_Higher_Order(tau);
            [Qr,~] = qr(f.A);
            ZT{i} = Qr(:,1:d)*Qr(:,1:d)';
            %ZT{i} = T*T';
            %Pr = eye(D)-X(:,i)*X(:,i)'/(norm(X(:,i))^2);
            %re_tangent(k,s) = re_tangent(k,s)+norm(P-Pr,'fro')^2;
            fprintf('the %d th sample\n',i);
       end
        %re(k,s) = norm(Z-T,'fro')^2;
        %fprintf('K=%d,s=%d,error=%.3f,error2=%.3f\n',k,s,re(k,s),re_tangent(k,s));
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
    data = data_true+sigma*(rand(3,num)-0.5);
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
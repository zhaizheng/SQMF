function Result = Projection_MovingLS(Data, inputs, neig, d)
    Result = [];
    for i = 1:size(inputs,2)
        x = inputs(:,i);
        h = find_sigma(x, Data, neig);
        [q, ~, ~, U] = find_origin(x, Data, h, d);
        [~,  c, A, B, R] = Least_Fitting(Data, U, q, h);
        tau = projection(x, A, B, c, rand(d,1));
        T = Construct_Higher_Order(tau); 
        Result = [Result, R*T];
    end
end

% function [c, A, P, Theta_h] = Least_Fitting(Data, Tau, h, q, rho, W)
% %   Tau = U'*(Data-q);
%     T= Construct_Higher_Order(Tau);
%     d = size(Tau, 1);
%     Theta_h = (build_theta(Data, h, q));
%     Theta = (Theta_h.^2)*W;
%     R = Construct_Regularization(Tau);
%     %R = Construct_Regularization2(d, T*Theta*T');
%     P = Data*Theta*T'/(T*Theta*T'+rho*R);
% end

function T = Construct_Higher_Order(Tau) 
    d = size(Tau, 1);
    T = zeros(1+d+d*(d+1)/2, size(Tau,2));
    ind = triu(true(size(Tau, 1)));
    for i = 1:size(Tau,2)
        T(1:1+d,i) = [1; Tau(:,i)];
        temp = Tau(:,i)*Tau(:,i)';
        T(d+2:end,i) = temp(ind);
    end 
end


function B = build_tensor(para, d)
    B = zeros(size(para,1),d,d);
    ind = triu(true(d));
    for i = 1:size(para,1)
        temp = zeros(d, d);
        temp(ind) = para(i,d+2:end);
        B(i,:,:) = (temp+temp')/2;
    end
end


function tau = projection(x, A, B, c, tau) %project x onto f(tau) = A tau+ B(tau,tau)+c
    iter = 0; 
    while true
        Bm = tensor_fold(B, tau);
        tau_new = (2*Bm'*Bm+Bm'*A+A'*A+A'*Bm)\((2*Bm'+A')*(x-c)-Bm'*A*tau);
        if norm(tau_new-tau)<1.e-6 || iter>300
            if iter>300
                fprintf('diverge projecting tau\n');
            end
            break;
        end
        tau = tau_new;
        iter = iter+1;
    end 
end


function result = tensor_fold(B, tau)
    result = zeros(size(B,1),size(B,2));
    for i = 1:size(B,1)
        result(i,:) = squeeze(B(i,:,:))*tau;
    end
end



function [q, k, Q, C] = find_origin(x, Data, h, d)
    q = x;
    U = principal(Data, h, q, d);
    k = 0;
    Q = [];
    while k<100
        Data_c = Data-repmat(q,1,size(Data,2));
        Theta = build_theta(Data, h, q);
        R = Data_c*Theta;
        Tau = U(:,1:d)'*R;
        Tau1 = [ones(1,size(Tau,2));Tau];
        A = R*Tau1'/(Tau1*Tau1');
        q_temp = q + A(:,1);
        [U,~] = qr(A(:,2:end));
        q_new = q_temp+U(:,1:d)*U(:,1:d)'*(x-q_temp);
        if norm(q_new-q)<0.001
            break;
        end
        q = q_new;
        Q = [Q,q];
        k = k+1;
    end
    C = U(:,1:d);
end


function [result, c, A, B, R] = Least_Fitting(Data, U, q, h)
    Tau = U'*(Data-q);
    T = Construct_Higher_Order(Tau);
    Theta = (build_theta(Data, h, q)).^2;
    R = Data*Theta*T'/(T*Theta*T');
    result = R(:,1);
    c = R(:,1);
    d = size(Tau,1);
    A = R(:,2:d+1);
    B = build_tensor(R, d);
end

% 
% function T = Construct_Higher_Order(Tau) 
%     d = size(Tau, 1);
%     T = zeros(1+d+d*(d+1)/2, size(Tau,2));
%     ind = triu(true(size(Tau, 1)));
%     for i = 1:size(Tau,2)
%         T(1:1+d,i) = [1; Tau(:,i)];
%         temp = Tau(:,i)*Tau(:,i)';
%         T(d+2:end,i) = temp(ind);
%     end
% end



function theta = build_theta(Data, h, q)  
    theta = diag(sqrt(exp(-sum((Data-q).^2,1)/h^2)));
end


function [U, center] = principal(Data, h, q, d)
    Theta = (build_theta(Data, h, q)).^2;
    center = sum(Data*Theta, 2)/sum(diag(Theta));
    [V,~,~] = svd((Data-center)*Theta*(Data-center)');
    U = V(:,1:d);
end


function sigma = find_sigma(x, Data, k)
    s_distance = sum((Data-x).^2, 1);
    [~,ind] = sort(s_distance,'ascend');
    Neig = Data(:,ind(2:k+1)); 
    sigma = max(sqrt(sum((Neig-x).^2,1)));
end
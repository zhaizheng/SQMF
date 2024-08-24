function [Data_new_Constructed, Taus, iter, Error, Q_projection, rho] = quadratic_fit(Data, q, Data_p, Tau, h1, center, W)
    Error = [];
    iter = 1;
    Q_projection = [];
    while true
        rho = 0;
        [c, A, R, Theta] = Least_Fitting(Data, Tau, h1, center, rho, W);
        
        d = size(Tau, 1);
        B = build_tensor(R, d);
        
        Data_Constructed = R*Construct_Higher_Order(Tau);
        
        Tau_q = projection2(q, A, B, c, zeros(d,1));
        q_projection = R*Construct_Higher_Order(Tau_q);
        Q_projection = [Q_projection, q_projection];
        
        Tau_old = Tau;
        for i = 1:size(Data_p, 2)
            Tau(:,i) = projection2(Data_p(:,i), A, B, c, Tau(:,i));
        end
        Tau_ee = qrs(Tau);
        Tau = Tau_ee(1:d,:);
        Taus{iter} = Tau;
        
        Data_new_Constructed = R*Construct_Higher_Order(Tau);
        error = norm(Data_new_Constructed- Data_Constructed,'fro');
        error2 = norm(Tau'*Tau- Tau_old'*Tau_old,'fro');
        
       
        error3 = norm((Data_Constructed-Data)*Theta,'fro');
        Error = [Error; [error, error2, error3]];
        if error < 1.e-4 || iter>800
            break;
        end
        iter = iter+1;
    end   
end







function sigma = find_sigma(x, Data, k)
    s_distance = sum((Data-x).^2, 1);
    [~,ind] = sort(s_distance,'ascend');
    Neig = Data(:,ind(2:k+1)); 
    sigma = max(sqrt(sum((Neig-x).^2,1)));
end



function Tau = qrs(Tau)
    d = size(Tau, 1);
    [Q,~] = qr([ones(size(Tau, 2),1),Tau']);
    %Tau = size(Tau,2)*Q(:,2:d+1)';
    Tau = Q(:,2:d+1)';
end


function [U,center] = principal(Data, h, q, d)
    Theta = (build_theta(Data, h, q)).^2;
    center = sum(Data*Theta, 2)/sum(diag(Theta));
    [V,~,~] = svd((Data-center)*Theta*(Data-center)');
    U = V(:,1:d);
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



function tau = projection2(x, A ,B, c, tau)
    iter = 0;
    while true
        r = A+2*tensor_fold(B, tau);
        e = A*tau+tensor_fold(B,tau)*tau+c-x;
        g = 2*r'*e;
        H = 2*r'*r+4*tensor_fold_1(B, e); 
        tau = tau-(1/norm(H))*g;
        if norm(g) < 1.e-7 || iter>300
            break;
        end
        iter = iter+1;
    end
    %fprintf('diverge projecting iter=%d,|g|=%f\n',iter,norm(g));
    function result = tensor_fold_1(B, e)
        result = zeros(size(B,2));
        for i = 1:size(B, 1)
            result = result + squeeze(B(i,:,:))*e(i);
        end
    end
end


function result = tensor_fold(B, tau)
    result = zeros(size(B,1),size(B,2));
    for i = 1:size(B,1)
        result(i,:) = squeeze(B(i,:,:))*tau;
    end
end



function [c, A, P, Theta_h] = Least_Fitting(Data, Tau, h, q, rho, W)
%   Tau = U'*(Data-q);
    T= Construct_Higher_Order(Tau);
    d = size(Tau, 1);
    Theta_h = (build_theta(Data, h, q));
    Theta = (Theta_h.^2)*W;
    R = Construct_Regularization(Tau);
    %R = Construct_Regularization2(d, T*Theta*T');
    P = Data*Theta*T'/(T*Theta*T'+rho*R);
    c = P(:,1);
    A = P(:,2:d+1);
end


function [g, t1, t2] = Eval_Rho(Data, Tau, h,q, rho)
    T= Construct_Higher_Order(Tau);
    d = size(Tau, 1);
    Theta_h = (build_theta(Data, h, q));
    Theta = Theta_h.^2;
    J = Construct_Regularization(Tau);
    R = Data*Theta*T'/(T*Theta*T'+rho*J);
    g = - trace(R*J/(T*Theta*T'+rho*J)*J*R'); 
    t1 = norm(R*J,'fro');
    t2 = norm(R*T-Data,'fro');
end


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


function R = Construct_Regularization(Tau)

    d = size(Tau, 1);
    R = zeros(1+d+d*(d+1)/2);
    R(d+2:end,d+2:end) = eye(d*(d+1)/2);

end





function theta = build_theta(Data, h, q)  
    theta = diag(sqrt(exp(-sum((Data-q).^2,1)/h^2)));
    %theta = diag(ones(1,size(Data, 2)));
end


function [data_true, data] = generate_data(sigma, num)
    theta = linspace(-pi/2, pi/2, num);%pi/4:0.1:3*pi/4;
    data_true = [cos(theta);sin(theta)];
    data = data_true+sigma*randn(2,length(theta));
end


function [data, data_t] = build_circle(sigma, num)
    theta = linspace(-pi, pi, num);
    data_t = [cos(theta);sin(theta)];
    data = data_t + sigma*randn(size(data_t));
end


function [data,data_t] = build_sin(sigma, num)
    theta = linspace(-pi, pi, num);
    data_t = [cos(theta);(theta)];
    data = data_t + sigma*randn(size(data_t));
end

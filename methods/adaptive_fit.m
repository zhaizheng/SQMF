function Result = adaptive_fit(Data, inputs, W, neig, n, d, delta)
    %k = floor(neig/2);
    k = neig;
    %n = neig;
    Result = [];
    for i = 1:size(inputs,2)
        fprintf('implement number :%d\n',i);
        q = inputs(:,i); 
        [Tau_p, Data_p, h, center, Weight] = initial_Tau(q, d, Data, k, n, W);  
        search.interval_l=0.001; search.interval_r=1;  search.epsilon = 0.001;  search.delta = delta;
        [~, ~, ~, ~, q_p, ~] = adaptive_fit_nonlinear(Data_p, q, Data_p, Tau_p, h, center, search, Weight);
        Result = [Result, q_p(:,end)];
    end
    
end


function [Tau, Data_selection, h, center, Weight] = initial_Tau(q, d, Data, k, n, W)
    [~,ind] = sort(sum((Data-q).^2,1),'ascend');
    Data_selection = Data(:,ind(2:n+1));
    Weight = W(ind(2:n+1),ind(2:n+1));
    
    h = find_sigma(q, Data, k);
    [U, center] = principal(Data_selection, h, q, d);
    Tau = U'*(Data-center);
   
    Tau_p = Tau(:,ind(2:n+1));
    Tau = qrs(Tau_p);
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


function [U, center] = principal(Data, h, q, d)
    Theta = (build_theta(Data, h, q)).^2;
    center = sum(Data*Theta, 2)/sum(diag(Theta));
    [V,~,~] = svd((Data-center)*Theta*(Data-center)');
    U = V(:,1:d);
end


function theta = build_theta(Data, h, q)  
    theta = diag(sqrt(exp(-sum((Data-q).^2,1)/h^2)));
    %theta = diag(ones(1,size(Data, 2)));
end

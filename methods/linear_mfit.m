
function [new_points_linear,ZT] = linear_mfit(Data, points, d, k)
        new_points_linear = zeros(size(points));
        ZT = cell(1,size(points,2));
        for i = 1:size(points,2)
            data_old = zeros(size(points(:,i)));
            data_move = points(:,i);
            sk = 1;
            while norm(data_old-data_move)>10.e-7 && sk<50
                data_old = data_move;               
%                 x = shift_mean(data_move, Data, sigma, k);
%                 [~, ~, ~, ~, U_v] = fitting(data_move, Data, sigma, D, d);
%                 data_move = data_old + U_v*U_v'*(x-data_old);
                step = 0.5; x = data_old; beta = 0.1;  data = Data;
                r = find_sigma(data_move, Data, k);
                [direction, P] = mfit(x, data, r, beta, d, step);
    %            direction = xia(x, data, r, beta, d, step);              
                data_move = data_old + step*direction;              
                sk = sk+1;
            end
            new_points_linear(:,i) = data_move;
            ZT{i} = P;
        end
end



function sigma = find_sigma(x, Data, k)
    s_distance = sum((Data-x).^2, 1);
    [~,ind] = sort(s_distance,'ascend');
    Neig = Data(:,ind(2:k+1)); 
    sigma = max(sqrt(sum((Neig-x).^2,1)));
end


function [direction, P] = mfit(x, data, r, beta, dim, step)
        d = size(data, 1);
        n = size(data, 2);
        d2 = 1 - sum((data-x).^2, 1)./(r^2);
        d2(d2<0) = 0;
        alpha_tilde = d2.^beta;
        alpha_sum = sum(alpha_tilde(:));
        alpha = alpha_tilde./alpha_sum;
        %cx = sum(data.* alpha, 2);
        Ns = zeros(d);
        c_vec = zeros(d,1);
        for i = 1:n
            if d2(i)>0
                ns = normal_space(data, i, r, dim);
                Ns = Ns+ns*alpha(i);
                c_vec = c_vec+alpha(i)*ns*(data(:,i)-x);
            end    
        end
        [U,~,~] = svd(Ns);
        P = U(:,1:d-dim)*U(:,1:d-dim)';
        direction = step*P*c_vec;
end


function P = normal_space(data, i, r, dim)
    ds = sum((data-data(:,i)).^2, 1);
    ds(ds > r^2) = 0;
    n = size(data, 2);
    d = size(data, 1);
    indicator = zeros([1,n]);
    indicator(ds>0) = 1;
    select = (data-data(:,i)).*indicator;
    cor = select*select';
    [U,~,~] = svd(cor);
    P = eye(d)-U(:,1:dim)*U(:,1:dim)';     
end

function direction = xia(x, data, r, beta, dim, step)
    d = size(data, 1);
    n = size(data, 2);
    d2 = 1 - sum((data-x).^2, 1)./(r^2);
    d2(d2<0) = 0;
    alpha_tilde = d2.^beta;
    alpha_sum = sum(alpha_tilde(:));
    alpha = alpha_tilde./alpha_sum;
    cx = sum(data.* alpha, 2);
    Ns = zeros(d);
    for i = 1:n
        if d2(i)>0
            ns = normal_space(data, i, r, dim);
            Ns = Ns+ns*alpha(i);
        end    
    end
    [U,~,~] = svd(Ns);
    P = U(:,1:d-dim)*U(:,1:d-dim)';
    direction = step*P*(cx - x );
end


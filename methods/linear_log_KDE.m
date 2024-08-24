function [new_points_linear,ZT] = linear_log_KDE(k, Data, points, d)
        D = size(Data, 1);
        new_points_linear = zeros(size(points));
        ZT = cell(1,size(points,2));
        for i = 1:size(points,2)
            data_old = zeros(size(points(:,i)));
            data_move = points(:,i);
            sk = 1;
            while norm(data_old-data_move)>10.e-7 && sk<50
                data_old = data_move;
                sigma = find_sigma(data_move, Data, k);
                x = shift_mean(data_move, Data, sigma, k);
                [~, ~, ~, ~, U_v] = fitting(x, Data, sigma, D, d);
                data_move = data_old + U_v*U_v'*(x-data_old);
                sk = sk+1;
            end
            ZT{i} = U_v*U_v';
            new_points_linear(:,i) = data_move;
        end
end


function sigma = find_sigma(x, Data, k)
    s_distance = sum((Data-x).^2, 1);
    [~,ind] = sort(s_distance,'ascend');
    Neig = Data(:,ind(2:k+1)); 
    sigma = max(sqrt(sum((Neig-x).^2,1)));
end


function [Theta, Co_h, Co_v, U_h, U_v] = fitting(x, Data, sigma, D, d)

    %mean = shift_mean(x, Data, sigma, k);
    [Co_h, Co_v, U_h, U_v] = coordinate(x, Data, sigma, D, d);
    W = build_W(x, Data, sigma);
    [~, Theta] = least_square(Co_h, Co_v, W);
end


function mean = shift_mean(x, Data, h, k)
    [~,ind] = sort(sum((Data-x).^2,1),'ascend');
    mean = zeros(size(x));
    s_weight = 0;
    for i = 1:k
        w = exp(-norm(Data(:,ind(i))-x)^2/(h^2));
        %w = 1;
        s_weight = s_weight+w;
        mean = mean+w*Data(:,ind(i));
    end
    mean = mean/s_weight;
end


function W = build_W(x, A, h)
    centered = A-repmat(x,[1,size(A,2)]);
    W = zeros(size(centered,2));
    for k = 1:size(centered,2)
        W(k,k) = exp(-norm(centered(:,k))^2/(h^2));
    end
end

function [theta, Theta] = least_square(Tau, Co_v, W)

    d = size(Tau, 1);
    ind = triu(true(size(Tau, 1)));
    d2 = d*(d+1)/2;
    Theta = cell(1,size(Co_v,1));
    for j = 1:size(Co_v, 1)
        G = zeros(d2, size(Tau,2));
        for i = 1:size(Tau,2)
            A = Tau(:,i)*Tau(:,i)';
            G(:,i) = A(ind);
        end
        theta = (G*W*G')\G*W*Co_v(j,:)';

        Theta_temp = zeros(d,d);
        Theta_temp(ind) = theta/2;
        Theta{j} = Theta_temp+Theta_temp';
    end
end


function [Co_h, Co_v, U_h, U_v] = coordinate(x, A, h, D, d)
        C = zeros(size(x,1),size(x,1));
        for i = 1: size(A,2)
            a = (x-A(:,i));
            C = C + exp(-norm(a)^2/(h^2))*a*a';
        end
        [V,~,~] = svd(C);
        U_h = V(:,1:d);
        U_v = V(:,d+1:D);
        Co_h = U_h'*(A-repmat(x,1,size(A,2)));
        Co_v = U_v'*(A-repmat(x,1,size(A,2)));
end

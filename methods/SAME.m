function X = SAME(Y, neig, d, K, alpha, gamma)
    X = zeros(size(Y));
    v = sum(Y.^2,1);
    n = size(Y, 2);
    D = v+v'-2*Y'*Y;
    SD = sort(D,1);
    hk = sqrt(mean(SD(neig,:)));
    tau = sqrt(mean(SD(floor(1.5*neig),:)));
    PI = compute_inital_PI(X, neig, d);
    for k = 1:K
        for i = 1:n
            pY_i = PI{i}*(Y - Y(:,i));
            W(i,:) = exp(-sum(pY_i.^2)/(hk^2));
        end
        W = W.*Indicator_Y(Y, tau);
        for i = 1:n
            X(:,i) = Y*W(i,:)'/sum(W(i,:));
        end
        PI = compute_PI(X, gamma*hk, d);
        hk = hk*alpha;
    end
end


function R = Indicator_Y(Y, tau)
    v = sum(Y.^2,1);
    D = v+v'-2*Y'*Y;
    R = D<(tau^2);
end


function PI = compute_inital_PI(X, neig, d)
    n = size(X,2);
    PI = cell(1,n);
    for i = 1:n
        cX = X-X(:,i);
        [~,ind] = sort(cX.^2,'ascend');
        Sigma_i = X(:,ind(2:neig+1))*X(:,ind(2:neig+1))';
        [U,~,~] = svd(Sigma_i);
        PI{i} = U(:,1:d)*U(:,1:d)';
    end
end


function PI = compute_PI(X, r, d)
    n = size(X, 2);
    PI = cell(1,n);
    for i = 1:n
        c = X-X(:,i);
        ind = sum(c.^2,1)<r^2;
        Cov = (X(:,ind)-X(:,i))*(X(:,ind)-X(:,i))';
        [U,~,~] = svd(Cov);
        PI{i} = U(:,1:d)*U(:,1:d)';
    end
end


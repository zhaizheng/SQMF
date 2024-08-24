function [Output,radius,C, ind] = Spherelet(DATA_TOBE, Data, k, d)
    Output = zeros(size(DATA_TOBE));
    C = zeros(size(DATA_TOBE));
    radius = zeros(1,size(DATA_TOBE,2));
    for i = 1:size(Output,2)
        [~, ind] = sort(sum((Data-DATA_TOBE(:,i)).^2,1),'ascend');
        S = Data(:,ind(2:k+1));
        c1 = mean(S,2);
        [U,~,~] = svds(double(S-c1), d+1);
        Y = c1+U*U'*(S-c1);
        [c,r] = compute_spherelet_center(Y);
        PU = U*U'*(DATA_TOBE(:,i)-c);
        res = c + r*PU/(norm(PU)+eps);
        if isnan(res(1))
            pause;
        end
        Output(:,i) = res;
        radius(i) = r;
        C(:,i) = c;
    end
end

% 
% function [c,r] = compute_spherelet_center(Y)
% 
%     c = mean(Y,2);
%     r = sqrt(mean(sum((Y-c).^2,1)));
%     while norm(center(Y,r)-c)>1.e-6
%         c = center(Y,r);
%         r = sqrt(mean(sum((Y-c).^2,1)));
%         norm(center(Y,r)-c)
%     end
% end
% 
% 
% function c = center(Y, r)
%     c = zeros(size(Y,1),1);
%     w = 0;
%     for i = 1:size(Y,2)
%         c = c + (norm(Y(:,i)-c)^2-r^2)*Y(:,i);
%         w = w + (norm(Y(:,i)-c)^2-r^2);
%     end
%     c = c/w;
% end

function [c,r] = compute_spherelet_center(Y)
    center = mean(Y,2);
    H = (Y-center)*(Y-center)';
    center2 = mean(diag(Y'*Y));
    xi = zeros(size(Y,1),1);
    for j = 1:size(Y,2)
        xi = xi + (Y(:,j)'*Y(:,j)-center2)*(Y(:,j)-center);
    end
    c = pinv(H)*xi/2;
    r = 0;
    n = size(Y,2);
    for j = 1:size(Y,2)
        r = r+norm(Y(:,j)-c)^2/n;
    end
    r = sqrt(r); 
    if isnan(r)
        pause;
    end
end
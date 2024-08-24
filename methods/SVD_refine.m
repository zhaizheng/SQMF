function Y_out = SVD_refine(Data_in, Data, k, d)
    Y_out = zeros(size(Data_in));
    for i = 1:size(Y_out,2)
        [~,ind] = sort(sum((Data-Data_in(:,i)).^2,1),'ascend');
        S = Data(:,ind(2:k+1));
%         center = mean(S,2);
%         S = S-center;
        [U,~] = svds(S, d);
        Y_out(:,i) = U*U'*Data_in(:,i);%+center;
    end
end
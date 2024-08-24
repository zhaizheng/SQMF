function [Y_out,ZT] = PCA_refine(Data_in, Data, k, d)
    Y_out = zeros(size(Data_in));
    ZT = cell(1,size(Y_out,2));
    for i = 1:size(Y_out,2)
        [~,ind] = sort(sum((Data-Data_in(:,i)).^2,1),'ascend');
        S = Data(:,ind(2:k+1));
        %center = mean(S,2);
        %S = S-center;
        [U, ~] = svds(S, d);
        Y_out(:,i) = U*U'*Data_in(:,i);
        ZT{i} = U*U';
    end
end



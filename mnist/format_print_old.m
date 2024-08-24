function format_print_old(result)
    [m,n] = size(result);
    for i = 1:m
        for j = 1:n
            fprintf('%s %.3f ','&',result(i,j));
        end
        fprintf('%s','\\');
        fprintf('\n')
    end
end
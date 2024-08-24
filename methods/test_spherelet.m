X = build_sphere(0.1, 240);
d = 1; D = 2;
K = 30;
for i = 1
    k = K(i);
    [Output, R, C,ind] = Spherelet(X(:,1), X, k, d);
    [result(i),~] = measure_distance(Output, projection(Output));
    
end
result

for j = 1%:240
    theta = 0:0.1:2.1*pi;
    x = R(j)*cos(theta);
    y = R(j)*sin(theta);
    plot(x+C(1,j),y+C(2,j),'-','LineWidth',4)
    hold on
end
hold on
plot(X(1,:),X(2,:),'>','MarkerSize',6,'MarkerFaceColor','g')
hold on
plot(X(1,ind(1:30)),X(2,ind(1:30)),'bo','MarkerSize',8,'MarkerFaceColor','r')
hold on

x1 = cos(theta);
y1 = sin(theta);
plot(x1,y1,'--','LineWidth',4)
axis off
set(gca,'FontSize',14)

function re = projection(A)
    re = bsxfun(@rdivide,A,sqrt(sum(A.^2,1)));
end


function [dis, s] = measure_distance(A, T)
    dis = norm(A-T,'fro')^2/size(A,2);
    S = A-T;
    s = std(sum(S.^2,1),1);
end


function data = build_sphere(sigma, num)

    data = randn(2,num);
    data = bsxfun(@rdivide,data, sqrt(sum(data.^2,1)));
    data = data + sigma*randn(size(data));
end
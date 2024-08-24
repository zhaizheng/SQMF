
[~, X] = generate_data3(0.1,150);
[~,ind] = find(rand(1,150)<0.4);
%%
t = tiledlayout(1,3,'TileSpacing','Compact');
nexttile
im1 = plot3(X(1,:),X(2,:),X(3,:),'.','MarkerSize',15,'Color','b');
%saveas(im1,'original_data.eps')
axis([-1,1,-1,1,-1,1])
%set(gca,'FontSize',18)
nexttile
demo_sphere(2,X,ind)
%saveas(im3,'quadratic_sphere.eps')
axis([-1.1,1.1,-1.1,1.1,-1.1,1.1])
%set(gca,'FontSize',18)
nexttile
demo_sphere(1,X,ind)
%saveas(im2, 'tangent_sphere.eps')
axis([-1.1,1.1,-1.1,1.1,-1.1,1.1])
%set(gca,'FontSize',18)
function demo_sphere(type,X,ind)
    k = 30;
    for i = 1:length(ind)
            Y = find_nearest(X(:,ind(i)),X, k);
            d = 2;
            [Q, x0, Theta, Phi, re] = Factorization(Y,d);
            phi1 = -0.3:.05:0.3; %min(Phi):(max(Phi)-min(Phi))/20:max(Phi);
            phi2 = -0.3:.05:0.3; 
            [X1,Y1] = meshgrid(phi1,phi2);
            Phi = [X1(:)';Y1(:)'];
            [psi, M] = Psi(Phi, Theta);
            if type == 1
                Z = x0 + Q(:,1:d)*Phi;
            elseif type == 2
                Z = x0 + Q*M;
            end
            r = size(X1,1);
            C = zeros(r,r,3);
            C(:,:,1) = ones(r);
            mesh(reshape(Z(1,:),r,[]),reshape(Z(2,:),r,[]),reshape(Z(3,:),r,[]),C);
            hold on
            %plot(Z(1,:),Z(2,:),'-','linewidth',4)
    end
    % if type == 2
    %     hold on
    %     plot3(X(1,:),X(2,:),X(3,:),'.','MarkerSize',15,'Color','b')
    % end
end



function demo_circle()
    [~, X] = generate_data(0.1,100);
    plot(X(1,:),X(2,:),'d','MarkerSize',6,'Color','r')
    k = 28;
    for i = 1:size(X,2)
        if mod(i,9) == 1
            Y = find_nearest(X(:,i),X, k);
            [Q, x0, Theta, Phi, re] = Factorization(Y);
            phi = -0.3:.05:0.3; %min(Phi):(max(Phi)-min(Phi))/20:max(Phi);
            [psi, M] = Psi(phi, Theta);
            Z = x0 + Q*M;
            hold on
            plot(Z(1,:),Z(2,:),'-','linewidth',4)
        end
    end
end


function Y = find_nearest(x, X, k)
    d = sum((X-x).^2,1);
    [~,ind] = sort(d,'ascend');
    Y = X(:,ind(2:k+1));
end


function test1()
    c = -1:0.1:1;
    y = c.^2/10;
    X = [cos(c);sin(c)]+0.1*rand(2,length(c));
    subplot(1,2,1)
    plot(X(1,:),X(2,:),'*');
    [Q, x0, Theta, Phi, re] = Factorization(X);
    hold on
    phi = min(Phi):(max(Phi)-min(Phi))/20:max(Phi);
    [psi, M] = Psi(phi, Theta);
    Z = x0 + Q*M;
    plot(Z(1,:),Z(2,:),'-*');
    subplot(1,2,2)
    plot(re)
end


function data = build_data()
    [data_true, data1] = generate_data(0.03, 50);
    [data_true, data2] = generate_data(0.03, 50);
    data = [[data1(1,:)+0.6*ones(size(data1(1,:)));data1(2,:)], data2];
end


function [data_true, data] = generate_data(sigma, num)
    theta = linspace(-2*pi, 2*pi, num);%pi/4:0.1:3*pi/4;
    data_true = [cos(theta);sin(theta)];
    data = data_true+sigma*randn(2,length(theta));
end


function [data_true, data] = generate_data3(sigma, num)
    rdata = rand(3,num)-0.5;
    E = diag(1./sqrt(sum(rdata.^2,1)));
    data_true = rdata*E;
    data = data_true+sigma*(rand(3,num)-0.5);
end


function [psi, M] = Psi(Phi, Theta)
    d = size(Phi,1);
    n = size(Phi,2);
    psi = [];%zeros(d*(d+1)/2,n);]
    for i = 1:d
        for j = i:d
            psi = [psi; Phi(i,:).*Phi(j,:)];
        end
    end
    M = [Phi; Theta'*psi];
end


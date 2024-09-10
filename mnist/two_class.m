clear
addpath('../')
data = load('mnist.mat'); 
Y  = data.trainY;
IND = [];
classes = [4,9];
C = [];
for i = 1:2
    ttt = find(Y==classes(i));
    IND = [IND, ttt(1:150)];
    C = [C,i*ones(1,150)];
end


%%
X = double(data.trainX)'; 
X = X(:,IND);

q = mean(X,2);
neig = 295;
d = 3;
rho = 0; k = neig; n = neig; delta = 0;

%[Tau_p, Data_p, ~, ~] = initial_Tau(q, d, X, k, n);  
%%
t0 = cputime;
Xc = X-mean(X,2);
[~, ~, V] = svd(Xc);
inital_tau = qrs(V(:,1:d)');
%%

%%
%[f,~] = fit_nonlinear(X, Tau, rho, delta, 0);  
%[Q_F, x0_F, Theta_F, Tau_F, error] = Factorization(X,d);
t1 = cputime;
total  = 100; alg = 3; lambda = 0; nd = 4;
[Q_F, x0_F, Theta_F, Tau_F, error] = Factorization3(X,d,total,alg,lambda, nd);
t2 = cputime;


[f_qmf,~] = RQMF(X,inital_tau,0.01,0);
t3 = cputime;

Xm = X-mean(X,2);
[U, ~, V] = svd(Xm);
Rc = U(:,1:d)*U(:,1:d)'*Xm+mean(X,2);
t4 = cputime;

[NMF_W,NMF_H] = FroNMF(X,d);
NMF_X = NMF_W *NMF_H;
t5 = cputime;

maxiter  = 1000;
[semi_W,semi_H] = semiNMF(X,d,maxiter);
semi_X = semi_W *semi_H;

t6 = cputime;
Eliptime = [t5-t4,t6-t5,t4-t3,(t3-t2)/length(f_qmf.data_error),(t2-t1)/length(error)];
%%
figure;
la = {'4','9'};

%la = {'0','1','2','3','4','5','6','7','8','9'};
% for i = 1:2
%     scatter3(Tau(1,C==i),Tau(2,C==i),Tau(3,C==i),50,'filled');
%     hold on
% end
% subplot(1,3,1)
% for i = 1:2
%     scatter(Tau(1,C==i),Tau(2,C==i),50,'filled');
%     hold on
% end
Tau = inital_tau;
t = tiledlayout(1,4,'TileSpacing','compact');
nexttile
for i = 1:2
    scatter(Tau(2,C==i),Tau(3,C==i),50,'filled');
    hold on
end
set(gca,'FontSize',18)
nexttile
for i = 1:2
    scatter(Tau(1,C==i),Tau(3,C==i),50,'filled');
    hold on
end
set(gca,'FontSize',18)

%t = tiledlayout(1,2,'TileSpacing','compact');
Tau2 = Tau_F;
%scatter(embedding(1,:),embedding(2,:),100,C,'filled');
%la = {'0','1','2','3','4','5','6','7','8','9'};
% subplot(1,3,1)
% for i = 1:2
%     scatter(Tau2(1,C==i),Tau2(2,C==i),50,'filled');
%     hold on
% end
nexttile
for i = 1:2
    scatter(Tau2(2,C==i),Tau2(3,C==i),50,'filled');
    hold on
end
set(gca,'FontSize',18)
nexttile
for i = 1:2
    scatter(Tau2(1,C==i),Tau2(3,C==i),50,'filled');
    hold on
end
set(gca,'FontSize',18)
legend(la);



%%






% figure
% 
% embedding = Tau_F;
% maxh = max(embedding(1,:));
% minh = min(embedding(1,:));
% hm = maxh-minh;
% maxv = max(embedding(2,:));
% minv = min(embedding(2,:));
% vm = maxv-minv;
% box off
% 
% %axis([minh-hm*0.15, maxh+hm*0.15,minv-vm*0.15, maxv+vm*0.15 ])
% s = 0.8/(max(embedding(1,:))-min(embedding(1,:)));
% c = 0.6/(max(embedding(2,:))-min(embedding(2,:)));
% smin = min(embedding(1,:));
% cmin = min(embedding(2,:));
% set(gca,'FontSize',18)
%%
% figure
% axis off
% for i = 1:size(embedding,2)
%     hold on   
%     if rand(1)>0.7
%         axes('Position',[(embedding(1,i)-smin)*s+0.1,(embedding(3,i)-cmin)*c+0.15,.07,.07]);
%         %subplot('Position',[embedding(1,i),embedding(2,i),.1,.1])
%         %Img_vec = f.Parm*Construct_Higher_Order(embedding(:,i));
%         [~, M_F] = Psi(Tau_F(:,i), Theta_F);
%         IMG  = x0_F + Q_F*M_F;
%         image(uint8(reshape(IMG,[28,28]))')
%         axis off
%     end
% end
%%



%%
figure;
t = tiledlayout(1,5,'TileSpacing','compact');
d1 = 1; d2 = 3;
nexttile
scatter(NMF_H(d1,:),NMF_H(d2,:),[],C,'filled');
set(gca,'FontSize',14)
nexttile
scatter(semi_H(d1,:),semi_H(d2,:),[],C,'filled');
set(gca,'FontSize',14)
nexttile
scatter(Pca(d1,:),Pca(d2,:),[],C,'filled');
set(gca,'FontSize',14)
nexttile
scatter(f_qmf.Tau(d1,:),f_qmf.Tau(d2,:),[],C,'filled');
set(gca,'FontSize',14)
nexttile
scatter(Tau2(d1,:),Tau2(d2,:),[],C,'filled');
set(gca,'FontSize',14)
%%
figure
position = [NMF_H(d1,:);NMF_H(d2,:)];
Image = NMF_X;
show_image_postion_d2(position, Image);
axis off

figure
position = [semi_H(d1,:);semi_H(d2,:)];
Image = semi_X;
show_image_postion_d2(position, Image);
axis off

figure
position = [Pca(d1,:);Pca(d2,:)];
Image = Rc;
show_image_postion_d2(position, Image);
axis off

figure
position = [f_qmf.Tau(d1,:);f_qmf.Tau(d2,:)];
Image = f_qmf.Data_Constructed;
show_image_postion_d2(position, Image);
axis off

figure
position = [Tau2(d1,:);Tau2(d2,:)];
[~, M_F] = Psi(Tau2, Theta_F);
Image  = x0_F + Q_F*M_F;
show_image_postion_d2(position, Image);
axis off


%%
addpath('/home/zzh/Documents/BPMA_CODE')

AccNmi = zeros(5,2);
[AccNmi(1,1),AccNmi(1,2)] = rep_kmeans(NMF_H' , 2, C, 40);
[AccNmi(2,1),AccNmi(2,2)] = rep_kmeans(semi_H' , 2, C, 40);
[AccNmi(3,1),AccNmi(3,2)] = rep_kmeans(inital_tau' , 2, C, 40);
[AccNmi(4,1),AccNmi(4,2)] = rep_kmeans(Construct_Higher_Order(f_qmf.Tau)' , 2, C, 40);
[~, M_F] = Psi(Tau2, Theta_F);
[AccNmi(5,1),AccNmi(5,2)] = rep_kmeans(qrs(M_F)',2, C, 40);
AccNmi



%%
M = 18;
figure
t = tiledlayout(6,M,'TileSpacing','tight');
IND  = sort(randperm(300,M));

for i = 1:M
    nexttile
    image(uint8(reshape(X(:,IND(i)),[28,28]))')
    axis off
end

for i = 1:M
    nexttile
    image(uint8(reshape(NMF_X(:,IND(i)),[28,28]))')
    axis off
end

for i = 1:M
    nexttile
    image(uint8(reshape(semi_X(:,IND(i)),[28,28]))')
    axis off
end

for i = 1:M
    nexttile
    image(uint8(reshape(Rc(:,IND(i)),[28,28]))')
    axis off
end

for i = 1:M
    nexttile
    %tau_q = projection(X(:,IND(i)),f_qmf.A,f_qmf.B,f_qmf.c,zeros(d,1));
    %qmf_proj = f_qmf.Parm*Construct_Higher_Order(tau_q);
    qmf_proj = f_qmf.Parm*Construct_Higher_Order(f_qmf.Tau(:,IND(i)));
    image(uint8(reshape(qmf_proj,[28,28]))');
    axis off
end


for i = 1:M
    nexttile
    %[~, M_F] = Psi(embedding(:,IND(i)), Theta_F);
    [~, M_F] = Psi(Tau2(:,IND(i)), Theta_F);
    Img_vec  = x0_F + Q_F*M_F;
    %Img_vec = f.Parm*Construct_Higher_Order(embedding(:,IND(i)));
    image(uint8(reshape(Img_vec,[28,28]))')
    axis off
end
%%
RREE = zeros(5,18);
for i = 1:18
    k = IND(i);
    
    RREE(1,i) = norm(X(:,k)-NMF_X(:,k))^2; 
    RREE(2,i) = norm(X(:,k)-semi_X(:,k))^2; 
    RREE(3,i) = norm(X(:,k)-Rc(:,k))^2;
    tau_q = projection(X(:,k),f_qmf.A,f_qmf.B,f_qmf.c,zeros(d,1));
    qmf_proj = f_qmf.Parm*Construct_Higher_Order(tau_q);
    RREE(4,i) = norm(X(:,k)-qmf_proj)^2;

    [~, M_F] = Psi(embedding(:,k), Theta_F);
    mg_vec  = x0_F + Q_F*M_F;
   % mg_vec = f.Parm*Construct_Higher_Order(embedding(:,k));
    RREE(5,i) = norm(X(:,k)-mg_vec)^2;
    %RREE(3,i) = (RREE(1,i)-RREE(2,i))/RREE(1,i);
end
mean(RREE,2)
%fprintf('the improvment for d=%d, is:%.3f\n', d, mean(RREE(3,:)))





%%
% 
% V = [0,1;1,0;1,1;1,-1];
% K = -.8:0.02:.8;
% K2 = -.7:.1:.7;
% fig = figure(1);
% for t = 1:4
%     if t == 1
%         K = K2;
%     end
%     for i = 1:length(K)
%         temp_coord = 0.2*V(t,:)'*(K(i));
%         Img_vec = f.Parm*Construct_Higher_Order(temp_coord);
%         %[~, M] = Psi(temp_coord, Theta);
%         %Moving_IMG  = x0 + Q*M;
%         axes('position',[(V(t,1)*K(i)+0.9)/2,(V(t,2)*K(i)+0.9)/2,.1,.1]);
%         imshow(uint8(reshape(Img_vec,[28,28]))')
%         hold on
%     end
% end
% saveas(fig,'two_class.eps');

%exportgraphics(ff, 'image.eps','Resolution', 300);


% function Result = quadratic(data, data_, k, n, d, delta ,W)
%         if ~exist('W','var')
%             W = diag(ones(1,size(data, 2))); %equal weight
%         end
%         Result = [];
%         for i = 1:size(data_,2)
%             q = data_(:,i);
%             [Tau_p, Data_p, ~, ~] = initial_Tau(q, d, data, k, n);  
%             rho = 0;
%             [f,~] = fit_nonlinear(Data_p, Tau_p, rho, delta, 0);        
%             %Tau_q = projection(q, f.A, f.B, f.c, zeros(d,1));
%             [Tau_q,~] = gradient_descent(q, f.A, f.B, f.c, zeros(d,1));
%             proj_q = f.Parm*Construct_Higher_Order(Tau_q);
%             Result = [Result,proj_q];
%             i
%         end      
% end

function S = normalize(A)
    n = size(A,1);
    [~,~,V] = svd(A);
    S = V(1:n,:);
end

function show_image_postion_d2(position, Image)
    embedding = position;
    s = 0.8/(max(embedding(1,:))-min(embedding(1,:)));
    c = 0.8/(max(embedding(2,:))-min(embedding(2,:)));
    smin = min(embedding(1,:));
    cmin = min(embedding(2,:));
    
    for i = 1:size(embedding,2)
        hold on   
        if rand(1)>0.7
            axes('Position',[(embedding(1,i)-smin)*s+0.1,(embedding(2,i)-cmin)*c+0.1,.06,.06]);
            %subplot('Position',[embedding(1,i),embedding(2,i),.1,.1])
            image(uint8(reshape(Image(:,i),[28,28]))')
            axis off
        end
    end
end



%%
function [Tau, Data_selection, h, center] = initial_Tau(q, d, Data, k, n)
    [~,ind] = sort(sum((Data-q).^2,1),'ascend');
    Data_selection = Data(:,ind(2:n+1));
    
    h = find_sigma(q, Data, k);
    [U, center] = principal(Data_selection, h, q, d);
    Tau = U'*(Data-center);
  
    Tau_p = Tau(:,ind(2:n+1));
    Tau = qrs(Tau_p);
end


function [f,rho] = fit_nonlinear(Data, Tau, rho, delta, adaptive, W, error)
    if ~exist('W','var')
        W = diag(ones(1,size(Data, 2))); %equal weight
    end
    if ~exist('error','var')
        error = 0.01;
    end
    iter = 1;
    f.q_sequence = [];
    while true
        % Parameter for regression Using Tau
        if adaptive == 1
            inter_l = 0.001; inter_r = 1; 
            rho = search_lambda(Data, Tau, inter_l, inter_r, inter_l, delta, W);
        end
        
        [c, A, P] = Least_Fitting(Data, Tau, rho, W);
        d = size(Tau, 1);
        B = build_tensor(P, d);
        
        f.Data_Constructed = P*Construct_Higher_Order(Tau);
        f.A = A; f.B = B; f.c = c;

        Tau_old = Tau;
        for i = 1:size(Data, 2)
            [Tau(:,i),~] = gradient_descent(Data(:,i), A, B, c, Tau(:,i));
            %Tau(:,i) = projection(Data(:,i), A, B, c, Tau(:,i));
        end
        Tau_ee = qrs(Tau);
        Tau = Tau_ee(1:d,:);
        f.Taus{iter} = Tau;
        f.Parm = P;
        f.Tau = Tau;
        f.Data_new_Constructed = P*Construct_Higher_Order(Tau);
        f.data_error = norm(f.Data_new_Constructed- f.Data_Constructed,'fro');
        f.Tau_error(iter) = norm(Tau'*Tau- Tau_old'*Tau_old,'fro');
        fprintf('error = %f,error1=%f,error2=%f\n',norm(f.Data_Constructed-Data,'fro')^2, f.data_error,f.Tau_error(iter));
        if f.Tau_error(iter) < error || iter>20 %1.e-4
            break;
        end
        iter = iter+1;
    end   
end



function theta = build_theta(Data, h, q)  
    theta = diag(sqrt(exp(-sum((Data-q).^2,1)/h^2)));
    %theta = diag(ones(1,size(Data, 2)));
end


function [c, A, P] = Least_Fitting(Data, Tau, rho, W)

    T= Construct_Higher_Order(Tau);
    d = size(Tau, 1);
    Theta = W.^2;
    R = Construct_Regularization(Tau);
    %R = Construct_Regularization2(d, T*Theta*T');
    P = Data*Theta*T'/(T*Theta*T'+rho*R);
    c = P(:,1);
    A = P(:,2:d+1);
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


function [U,center] = principal(Data, h, q, d)
    Theta = (build_theta(Data, h, q)).^2;
    center = sum(Data*Theta, 2)/sum(diag(Theta));
    [V,~,~] = svd((Data-center)*Theta*(Data-center)');
    U = V(:,1:d);
end


function B = build_tensor(para, d)
    B = zeros(size(para,1),d,d);
    ind = triu(true(d));
    for i = 1:size(para,1)
        temp = zeros(d, d);
        temp(ind) = para(i,d+2:end);
        B(i,:,:) = (temp+temp')/2;
    end
end


function [tau,Tau] = gradient_descent(x, A, B, c, tau)
    Tau = [];
    iter = 0;
    s = 1;
    while true
        Bm = tensor_fold(B, tau);
        v = x- (A*tau+ Bm*tau+c);
        g = (-A-2*Bm)'*v;
        while f_value(x, A, B, c,tau-s*g) > f_value(x, A, B, c, tau) 
            s = s/2;
        end
        tau_new = tau - s * g;
        if norm(tau_new-tau)<1.e-6 || iter>3000
            break;
        end
        tau = tau_new;
        Tau = [Tau, tau];
        iter = iter+1;
    end
end

function a = f_value(x, A, B, c, tau)
    Bm = tensor_fold(B, tau);
    a = norm(x-(A*tau+ Bm*tau+c))^2;
end


% function tau = projection(x, A, B, c, tau) %project x onto f(tau) = A tau+ B(tau,tau)+c
%     iter = 0; 
%     while true
%         Bm = tensor_fold(B, tau);
%         tau_new = (2*Bm'*Bm+Bm'*A+A'*A+A'*Bm)\((2*Bm'+A')*(x-c)-Bm'*A*tau);
%         if norm(tau_new-tau)<1.e-6 || iter>300
% %             if iter>300
% %                 fprintf('diverge projecting tau\n');
% %             end
%             break;
%         end
%         tau = tau_new;
%         iter = iter+1;
%     end 
% end

% 
% function [tau,Tau] = gradient_descent(x, A, B, c, tau)
%     Tau = [];
%     iter = 0;
%     while true
%         Bm = tensor_fold(B, tau);
%         v = x- (A*tau+ Bm*tau+c);
%         g = (-A-2*Bm)'*v;
%         tau_new = tau - 0.001 * g;
%         if norm(tau_new-tau)<1.e-6 || iter>1000
%             break;
%         end
%         tau = tau_new;
%         Tau = [Tau, tau];
%         iter = iter+1;
%     end
% end


function result = tensor_fold(B, tau)
    result = zeros(size(B,1),size(B,2));
    for i = 1:size(B,1)
        result(i,:) = squeeze(B(i,:,:))*tau;
    end
end






function T = Construct_Higher_Order(Tau) 
    d = size(Tau, 1);
    T = zeros(1+d+d*(d+1)/2, size(Tau,2));
    ind = triu(true(size(Tau, 1)));
    for i = 1:size(Tau,2)
        T(1:1+d,i) = [1; Tau(:,i)];
        temp = Tau(:,i)*Tau(:,i)';
        T(d+2:end,i) = temp(ind);
    end
   
end


function R = Construct_Regularization(Tau)

    d = size(Tau, 1);
    R = zeros(1+d+d*(d+1)/2);
    R(d+2:end,d+2:end) = eye(d*(d+1)/2);
    
    %R = eye(1+d+d*(d+1)/2);
end


function R = Construct_Regularization2(d, A)
    
    [U,~, ~] = svd(A);
    R = U(:,d+1:end)*U(:,d+1:end)';
    %R = eye(1+d+d*(d+1)/2);
end


% function theta = build_theta(Data, h, q)  
%     %theta = diag(sqrt(exp(-sum((Data-q).^2,1)/h^2)));
%     theta = diag(ones(1,size(Data, 2)));
% end


function [data_true, data] = generate_data(sigma, num)
    theta = linspace(-pi/2, pi/2, num);%pi/4:0.1:3*pi/4;
    data_true = [cos(theta);sin(theta)];
    data = data_true+sigma*randn(2,length(theta));
end


function data = build_circle(sigma, num)
    theta = linspace(-pi, pi, num);
    data = [cos(theta);sin(theta)];
    data = data + sigma*randn(size(data));
end

% function data = build_sphere(sigma, num)
% 
%     data = randn(3,num);
%     %data = data*diag(1./sqrt(sum(data.^2.1)));
%     data = bsxfun(@rdivide,data, sqrt(sum(data.^2,1)));
%     data = data + sigma*randn(size(data));
% end



function data = build_sin(sigma, num)
    theta = linspace(-pi, pi, num);
    data = [cos(theta);(theta)];
    data = data + sigma*randn(size(data));
end

function data = build_data()

    [data_true, data1] = generate_data(0.03, 50);

    [data_true, data2] = generate_data(0.03, 50);

    data = [[data1(1,:)+0.6*ones(size(data1(1,:)));data1(2,:)], data2];
end


function re = reshape_s(a)
    n = sqrt(length(a));
    re = reshape(a,[n,n]);
end


function re = ini(a,b,num,d)
    l1 = linspace(a,b,num);
    l2 = linspace(a,b,num);
    [A, B] = meshgrid(l1,l2);
    re = zeros(d,num*num);
    re(1,:) = A(:);
    re(2,:) = B(:);
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
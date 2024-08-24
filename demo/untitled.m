t = -pi/3:0.05:2*pi/3;
x = cos(t);
y = sin(t);
Data = [x;y];
addpath('./../')
d = 1;
Datan = cell(1,4);
for w = 1:5
    nData{w} = Data + w*0.1*randn(2,length(t));
end
%%
lambda = 0;
nd = 1;
 %1 Newton 2, gradient 3, surrogate
Error = cell(3,5);
Time = zeros(3,5);
figure
tl = tiledlayout(3, 5,'TileSpacing','compact');
for approach = 1:3
    for w = 1:5
        temp = nexttile;
        Datan = nData{w};
        plot(Datan(1,:),Datan(2,:),'*','MarkerSize',3)
        for K = 30
            t = cputime;
            [Q, x0, Theta, Tau, Error{approach,w}] = Factorization3(Datan, d, K, approach, lambda, nd);
            %fprintf('timecost for apporach %d =%f\n',approach,cputime-t);
            Time(approach,w) = cputime-t;
            [~, M] = Psi(-1.2:0.1:1.2, Theta);
            Fit = x0+Q*M;
            hold on
            plot(Fit(1,:),Fit(2,:),'b-','LineWidth',3)
            hold on
            [~, MT] = Psi(Tau, Theta);
            Fit_T = x0+Q*MT;
            plot(Fit_T(1,:),Fit_T(2,:),'o','MarkerSize',8)
        end
        axis([-1 2 -1 2])
        hold on
        for i = 1:size(Fit_T,2)
            plot([Datan(1,i),Fit_T(1,i)],[Datan(2,i),Fit_T(2,i)],'r--','LineWidth',1);
        end
        title(temp, {['$\sigma=',num2str(w*0.1),'$']}, 'Interpreter', 'latex', 'FontSize', 14);
    end
end
%%
figure
t2 = tiledlayout(1,5,"TileSpacing",'compact');
for w = 1:5
    temp = nexttile;
    for approach = 1:3
        hold on
        plot(Error{approach,w},'LineWidth',1.5);
    end
    legend({'Newton','Gradient','Surrogate'})
    title(temp, {['$\sigma=',num2str(w*0.1),'$']}, 'Interpreter', 'latex', 'FontSize', 14);
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

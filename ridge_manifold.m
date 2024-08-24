% x1 = linspace(-3,3,500);
% y1 = zeros(size(x1));
% 
% x2 = zeros(size(x1));
% y2 = [linspace(-3,-1,250),linspace(1,3,250)]
% D1 = [x1;y1]+0.6*randn(2,500);
% D2 = [x2;y2]+0.6*randn(2,500);
% plot(D1(1,:),D1(2,:),'*')
% hold on
% plot(D2(1,:),D2(2,:),'*')
Data = [];
D = (rand(2,5000)-0.5)*8;
for i = 1:5000
    t = rand(1);
    if t < 3*f(D(1,i),D(2,i))
        Data = [Data,D(:,i)]
    end
end

plot(Data(1,:),Data(2,:),'*')
axis([-2.5 2.5 -2.5 2.5])
hold on
x1 = linspace(-3,3,100);
y1 = zeros(size(x1));

x2 = zeros(size(x1));
y2 = [linspace(-3,-1,50),linspace(1,3,50)];
plot(x1,y1,'r-','LineWidth',2);
hold on
plot(x2(1:50),y2(1:50),'r-','LineWidth',2);
hold on
plot(x2(51:100),y2(51:100),'r-','LineWidth',2);
function u = f(x,y)
    u = exp(-x^2-3*y^2);
end
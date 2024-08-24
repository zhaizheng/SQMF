theta = linspace(0,2*pi,200);
real = [cos(theta);sin(theta)];
Data = real+0.04*randn(2,200);
plot(Data(1,:),Data(2,:),'*');
inputs = [0.6;0.2];
neig = 20;
d = 1;
hold on
for i = 1:size(Data,2)
    inputs = Data(:,i);
    out = PCA_refine(inputs, Data, neig, d); %linear_mfit(Data, inputs, d, neig);
    %re = MovingLS(Data, inputs, neig, d);
    %plot(re.result(1),re.result(2),'o');
    plot(out(1),out(2),'o');
    hold on
end
plot(real(1,:),real(2,:),'-')

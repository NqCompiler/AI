load('ex5data1.mat');

X = [ones(length(X),1) X];
[m, n] = size(X);
lambda = 1;

theta = [1; 1];
theta_reg = theta;
theta_reg(1) = 0;

h = X*theta;
J = 1/(2*m)*sum((h-y).^2)+(lambda/(2*m))*sum(theta_reg.^2)
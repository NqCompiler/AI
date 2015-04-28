function [ J, grad ] = lrCostFunction( theta, X, y, lambda )
%LRCOSTFUNCTION Summary of this function goes here
%   Detailed explanation goes here
[m, n] = size(X);
z = X*theta;
h = sigmoid(z);

theta_lambda = theta; %401x1
theta_lambda(1) = 0;

J = 1/(m)*sum(-y.*log(h)-(1-y).*log(1-h)) + lambda/(2*m)*sum(theta_lambda.^2);

grad = (1/m).*(X'*(h - y)) + lambda/m*(theta_lambda);

grad = grad(:);



end


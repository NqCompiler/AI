function [ J, grad ] = costFunction( theta, X, y, lambda )
%GRADIENTDESCENT Summary of this function goes here
%   Detailed explanation goes here

h = sigmoid(X*theta);
[m, n] = size(X);
J = 1/m*sum( -y.*log(h)-(1-y).*log(1-h) ) + lambda/(2*m)*sum(theta(2:end).^2);

grad = zeros(n,1);
grad(1) = 1/m*X(:,1)'*(h-y);
grad(2:end) = 1/m*X(:,2:end)'*(h-y) + lambda/m*theta(2:end);

end


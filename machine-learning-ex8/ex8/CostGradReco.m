function [ cost, grad ] = CostGradReco( X_Theta, y, R, lambda, num_features )
%COSTGRADRECO Summary of this function goes here
%   Detailed explanation goes here

[m, n] = size(y);
X      = reshape(X_Theta(1: m*num_features), m, num_features);
Theta  = reshape(X_Theta(m*num_features+1: end), n, num_features);


h = X*Theta';   % 1682x943

cost = 0.5*sum(sum((R.*(h-y)).^2)) + ...
    (lambda/2)*(sum(sum(Theta.^2)) + sum(sum(X.^2))); 

grad_x     = ((h-y).*R)*Theta + lambda*X; % 1682x10
grad_theta = ((h-y).*R)'*X + lambda*Theta; % 943x10 + 943x10

grad = [grad_x(:); grad_theta(:)];

end


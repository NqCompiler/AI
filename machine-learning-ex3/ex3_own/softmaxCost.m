function [ J, grad ] = lrCostFunction( theta_un, X, y, lambda )
%LRCOSTFUNCTION Summary of this function goes here
%   theta: n x K
K = max(y);
[m, n] = size(X);

y_k = zeros(m, K);
a = eye(K);
y_k(:, :) = a(y, :);

theta = reshape(theta_un(1:end), n, K);
z = X*theta;    % m x K (m x n * n x K)
h = exp(z);     % m x K
exp_sum = sum(h, 2);                % m x 1
h = bsxfun(@rdivide, h, exp_sum);   % f x u

theta_lambda = theta; % n x 1
theta_lambda(1) = 0;

J = - 1/(2*m)*sum(sum(y_k .* log(h))) + lambda/(2*m)*sum(sum(theta_lambda.^2));
grad = - 1/m*X'*(y_k - h) + lambda/m * theta_lambda;

grad = grad(:);



end


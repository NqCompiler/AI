function [ J, grad ] = nnCostFunction( theta_un, X, y, y_class, lambda )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

t1 = reshape(theta_un(1:25 * 401), 25, 401);      % 25 x 401
t2 = reshape(theta_un(25 * 401 + 1:end), 10, 26); % 10 x 26
num_labels = max(y);

% forward propagation

[m, n] = size(X);
a1 = X;
z2 = X*t1';                     % 5000 x 25
a2 = tanh(z2);                  % 5000 x 25
a2 = [ones(m,1) a2];            % 5000 x 26
z3 = a2*t2';
%h  = tanh(z3);                 % 5000 x 10
h_exp     = exp(z3);
h_exp_sum = sum(h_exp, 2);   % 5000 x 1
h_softmax = bsxfun(@rdivide, h_exp, h_exp_sum);

y_classd3 = y_class;
y_classd3(find(y_classd3==-1)) = 0;

J = 1/(m) * sum(sum((y_classd3 - h_softmax).^2)) + (lambda / (2 * m)) * (sum(sum(t1 .^ 2)) + sum(sum(t2 .^2 )));

% Backpropagation vectorized

d3 = - (y_classd3 - h_softmax).*(1-tanh(z3).^2); % 5000 x 10
d2 = (d3 * t2(:,2:end))' .* (1-tanh(z2).^2)';  % 25 x 5000
D2 = d3' * a2;                                 % 10 x 26
D1 = d2 * a1;                                  % 25 x 401

t1_reg = t1;
t1_reg(:,1) = 0;
t2_reg = t2;
t2_reg(:,1) = 0;
 
D1 = 1/m * D1 + (lambda/m) .* t1_reg;
D2 = 1/m * D2 + (lambda/m) .* t2_reg;
 
grad = [D1(:); D2(:)];

end


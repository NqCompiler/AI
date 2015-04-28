function [ J, grad ] = nnCostFunction( theta_un, X, y_class, lambda )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

t1 = reshape(theta_un(1:25 * 401), 25, 401);      % 25 x 401
t2 = reshape(theta_un(25 * 401 + 1:end), 10, 26); % 10 x 26

% forward propagation

[m, n] = size(X);
a1 = X;
z2 = X*t1';                  % 5000 x 25
a2 = sigmoid(z2);            % 5000 x 25
a2 = [ones(m,1) a2];         % 5000 x 26
h = sigmoid(a2*t2');         % 5000 x 10
% [dummy h_ind] = max(h,[],2); % 5000 x 10

J = 1/m * sum(sum( -y_class .* log(h) - (1 - y_class) .* log(1 - h) )) + (lambda / (2 * m)) * (sum(sum(t1 .^ 2)) + sum(sum(t2 .^2 )));

% Backpropagation vectorized

d3 = h - y_class;                        % 5000 x 10
d2 = (d3 * t2(:,2:end))' .* sigmoidG(z2)';% 25 x 5000
D2 = d3' * a2;                           % 10 x 26
D1 = d2 * a1;                            % 25 x 401

% Backpropagation unvectorized
% 
% D1 = zeros(size(t1));
% D2 = zeros(size(t2));
% 
% for t = 1:m
%     a1 = X(t,:);            % 1   x 401
%     z2 = t1 * a1';          % 25  x 1
%     a2 = sigmoid(z2);       % 25  x 1
%     a2 = [1; a2];           % 26  x 1
%     z3 = t2 * a2;           % 10  x 1
%     a3 = sigmoid(z3);       % 10  x 1
%     
%     d3 = a3 - y_class(t,:)';            % 10  x 1
%     d2 = t2' * d3 .* sigmoidG([1; z2]); % 26  x 1  
%     D2 = D2 + d3 * a2';                 % 10  x 26
%     D1 = D1 + d2(2:end,:) * a1;         % 25 x 401
% end

t1_reg = t1;
t1_reg(:,1) = 0;
t2_reg = t2;
t2_reg(:,1) = 0;
 
D1 = 1/m * D1 + (lambda/m) .* t1_reg;
D2 = 1/m * D2 + (lambda/m) .* t2_reg;
 
grad = [D1(:); D2(:)];

end


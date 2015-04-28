function [ J ] = nnForward( theta_un, X, y_class, lambda )
%NNFORWARD Summary of this function goes here
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

J = 1/m * sum(sum( -y_class .* log(h) - (1 - y_class) .* log(1 - h) ));

end


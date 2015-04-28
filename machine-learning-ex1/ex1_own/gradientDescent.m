function [ theta, J_his ] = gradientDescent( X, y, theta, alpha, iterations )
%GRADIENTDESCENT calculates the batch gradient descent of
% a given dataset.
m = length(y);
J_his = zeros(m, 1);

for i = 1:iterations
    J_his(i) = computeCost(X, y, theta);
    J_his(i);
    h = X*theta;
    theta = theta - alpha/m.*((h-y)'*X)';
end

end


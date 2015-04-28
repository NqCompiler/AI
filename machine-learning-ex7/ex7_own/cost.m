function [ J ] = cost( X,centroids,c )
%COST Summary of this function goes here
%   Detailed explanation goes here
[m n] = size(X);
J = 0;
for i = 1:max(c)
    J = J + sum(sum( bsxfun(@minus,X(find(c == i),:),centroids(i,:)) .^ 2));
end
J = (1 / m) * J;
end


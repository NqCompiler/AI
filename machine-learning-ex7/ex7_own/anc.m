function [ mu_c ] = anc( X, c, k )
%ANC Summary of this function goes here
%   Detailed explanation goes here
[m,n] = size(X);
mu_c = zeros(k,n);

for i = 1:k
   mu_c(i,:) = mean(X(find(c==i),:), 1);
end

end


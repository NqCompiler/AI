function [ c ] = fcc( X,centroids )
%FCC Summary of this function goes here
%   Detailed explanation goes here
[m, n] = size(X);
[m_c, n_c] = size(centroids);
c = zeros(m,1);

% standard code with repmat
for i = 1:m
   [val, c(i)] = min(sum((repmat(X(i, :), m_c, 1) - centroids) .^ 2, 2)); 
end

% code with bsxfun
%for i = 1:m
%   [val, c(i)] = min( sum( bsxfun(@minus, X(i, :), centroids).^2 , 2) );
%end

end


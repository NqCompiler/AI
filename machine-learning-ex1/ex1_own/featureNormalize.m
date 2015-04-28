function [ normalized ] = featureNormalize( X )
%FEATURENORMALIZE Summary of this function goes here
%   Detailed explanation goes here

mu = mean(X);
sigma = std(X);

% normalized = (X-repmat(mu,length(X),1))./repmat(sigma,length(X),1);

normalized = bsxfun(@minus, X, mu);
normalized = bsxfun(@rdivide, normalized, sigma);

end


function [ J ] = computeCost( X, y, theta )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
m = length(y);
h = X*theta;
J = 1/(2*m)*sum((h-y).^2);

end


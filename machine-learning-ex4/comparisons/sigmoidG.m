function [ h ] = sigmoidG( z )
%SIGMOIDG Summary of this function goes here
%   Detailed explanation goes here

g = sigmoid(z);
h = g.*(1-g);

end


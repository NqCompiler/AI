function [ h ] = sigmoid( z )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here

h = 1./ ( 1 + exp(-z) );

end


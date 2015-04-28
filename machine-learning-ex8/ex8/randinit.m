function [ out ] = randinit( m, max_features, epsilon )
%RANDINIT Summary of this function goes here
%   Detailed explanation goes here

out = rand(m, max_features)*2*epsilon - epsilon;

end


function [ W ] = randInitializeWeights( L_out, L_in )
%RANDINITIALIZEWEIGHTS Summary of this function goes here
%   Detailed explanation goes here

epsilon_init = sqrt(6)/sqrt(L_in + L_out);
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end


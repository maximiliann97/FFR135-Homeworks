%Maximilian Salén
%19970105-1576
%Last Updated: 2022-09-28
clear all
close all
clc

% Intialize XOR patterns
x = ...
    [-1 -1 -1;...
    1, -1, 1;...
    -1, 1, 1;...
    1, 1, -1;...
    1, 1, 1;...
    -1, -1, 1;...
    1, -1, -1;...
    -1, 1, -1];

% Initialize variables
N = 3;
M = [1 2 4 8];
eta = 0.01;
k = 200;


% Iteration variables
counter = 0;
nTrials = 100;


while true
    for trial = 1:nTrials
        dW = 0;
        dTheta_h = 0;
        dTheta_v = 0;
        for minibatch = 1:20
            mu = randi(4);

        end
    end
end









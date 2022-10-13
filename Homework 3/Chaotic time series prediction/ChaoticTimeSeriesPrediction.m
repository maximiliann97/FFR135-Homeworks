% Maximilian Salén
% 19970105-1576
% Last updated: 2022-10-13
clear all
close all
clc

%Load data
xTest = load("test-set-9.csv");
xTraining = load('training-set.csv');

%Initialize
N = 3; %input neurons
nReservoirNeurons = 500;
k = 0.01; %ridge parameter
w_in = normrnd(0, 0.002, [nReservoirNeurons, N]);
w = normrnd(0, 2/nReservoirNeurons, [nReservoirNeurons, nReservoirNeurons]);

tTrain = length(xTraining(1,:));
r = zeros(nReservoirNeurons,1);
R = zeros(nReservoirNeurons,tTrain);

% Train the reservoir
for t = 1:tTrain
    r = tanh(LocalField(w,r) + LocalField(w_in,xTraining(:,t)));
    R(:,t) = r;
end

w_out = xTraining*R'*inv(R*R' + k*eye(nReservoirNeurons));
O_i = LocalField(w_out,r);


% Prediction
tTest = length(xTest);
%r = zeros(nReservoirNeurons,1);

for t = 1:tTest
    r = tanh(LocalField(w,r) + LocalField(w_in,xTest(:,t)));
end

O = LocalField(w_out,r);
y = [];

for t = 1:500
    y = [y O];
    r = tanh(LocalField(w,r) + LocalField(w_in,O));
    O = LocalField(w_out,r);
end

csvwrite("prediction.csv",y(2,:))
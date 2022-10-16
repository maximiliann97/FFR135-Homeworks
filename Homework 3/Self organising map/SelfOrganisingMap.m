% Maximilian Salén
% 19970105-1576
% Last updated: 2022-10-14
clear all
close all
clc

% Load data
data = load('iris-data.csv');
data = data./max(data); %Normalize
labels = load('iris-labels.csv');


% Initialize
nEpochs = 10;
batchSize = 1;
nInputs = length(data);
eta = 0.1;    %Initial learning rate
d_eta = 0.01;   %with decay rate
sigma = 10;   %Initial width of neighbourhood function
d_sigma = 0.1;  %with decay rate
W = rand(40,40,4);
W_init = W;


for epoch = 1:nEpochs
    eta = eta * exp(-d_eta*epoch);
    sigma = sigma * exp(-d_sigma*epoch);
    for input = 1:nInputs
        randomIndex = randi(nInputs);
        X(1,1,:) = data(randomIndex,:);
        distance = zeros(40);
        for k = 1:length(X) 
            distance = distance + sqrt(W(:,:,k) - X(k)).^2;
        end
        [i_min,j_min]  = find(distance==min(distance(:)));
        r0 = [i_min j_min];

        for i = 1:height(distance)
            for j = 1:length(distance)
                r = [i j];
                distance_r0 = vecnorm(r-r0);
                
                if distance_r0 < sigma/3
                    h = exp(-(1/2*sigma^2) * distance_r0);
                    dW = eta*h*(X-W(i,j,:));
                    W(i,j,:) = W(i,j,:) + dW;
                end
            end
        end
    end
end


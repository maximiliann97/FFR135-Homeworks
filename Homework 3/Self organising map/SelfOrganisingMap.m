% Maximilian Salén
% 19970105-1576
% Last updated: 2022-10-16 
clear all
close all
clc

% Load data
data = load('iris-data.csv');
data = data./max(max(data)); %Normalize
labels = load('iris-labels.csv');


% Initialize
nEpochs = 10;
nInputs = length(data);
eta = 0.1;    %Initial learning rate
d_eta = 0.01;   %with decay rate
sigma = 10;   %Initial width of neighbourhood function
d_sigma = 0.05;  %with decay rate
W = rand(40,40,4);
W_init = W;

% Training
for epoch = 1:nEpochs
    eta = eta * exp(-d_eta*epoch);
    sigma = sigma * exp(-d_sigma*epoch);
    for input = 1:nInputs
        
        randomIndex = randi(nInputs);
        X(1,1,:) = data(randomIndex,:);
        termsInit = {zeros(40) zeros(40) zeros(40) zeros(40)};
        for k = 1:length(X)
            termsInit{k} = (W(:,:,k) - X(k)).^2;
        end
        distanceInit = sqrt(termsInit{1} + termsInit{2} + termsInit{3} + termsInit{4});
        [i_min,j_min]  = find(distanceInit==min(distanceInit(:)));
        r0 = [i_min j_min];

        for i = 1:height(distanceInit)
            for j = 1:length(distanceInit)
                r = [i j];
                distance_r0 = vecnorm(r-r0);
                if distance_r0 < 3*sigma
                    h = exp(-(1/2*sigma^2) * distance_r0);
                    dW = eta*h*(X-W(i,j,:));
                    W(i,j,:) = W(i,j,:) + dW;
                end
            end
        end
    end
end


for i = 1:nInputs
        X(1,1,:) = data(i,:);
    for n = 1:4
        termsInit{n} = (squeeze(W_init(:,:,n)) - X(n)).^2;
        termsFinal{n} = (squeeze(W(:,:,n)) - X(n)).^2;
    end
        noise = normrnd(0,0.02);
        noise2 = normrnd(0,0.02);
        distanceInit = sqrt(termsInit{1} + termsInit{2} + termsInit{3} + termsInit{4});
        [i_min,j_min]  = find(distanceInit==min(distanceInit(:)));
        initWinning_i(i) = i_min + noise;
        initWinning_j(i) = j_min + noise2;

        distanceFinal = sqrt(termsFinal{1} + termsFinal{2} + termsFinal{3} + termsFinal{4});
        [i_minF,j_minF]  = find(distanceFinal==min(distanceFinal(:)));
        finalWinning_i(i) = i_minF +  noise;
        finalWinning_j(i) = j_minF + noise2;
end

figure(1)
scatter(initWinning_i,initWinning_j,'b.')
figure(2)
scatter(finalWinning_i,finalWinning_j,'r.')
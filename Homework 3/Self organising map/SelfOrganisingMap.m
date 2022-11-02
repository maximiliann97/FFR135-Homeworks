% Maximilian Salén
% 19970105-1576
% Last updated: 2022-10-17
clear all
close all
clc

% Load data
data = load('iris-data.csv');
data = data./max(data); %Normalize
labels = load('iris-labels.csv');


% Initialize
nEpochs = 10;
nInputs = length(data);
eta = 0.1;    %Initial learning rate
d_eta = 0.01;   %with decay rate
sigma = 10;   %Initial width of neighbourhood function
d_sigma = 0.05;  %with decay rate
W = rand([40,40,4]);
W_init = W;
terms = {zeros(40) zeros(40) zeros(40) zeros(40)};

% Training
for epoch = 1:nEpochs
    eta = eta * exp(-d_eta*epoch);
    sigma = sigma * exp(-d_sigma*epoch);
    for input = 1:nInputs
        randomIndex = randi(nInputs);
        X(1,1,:) = data(randomIndex,:);

        for k = 1:length(X)
            terms{k} = (W(:,:,k) - X(k)).^2;
        end
        distance = sqrt(terms{1} + terms{2} + terms{3} + terms{4});

        % Updating weights
        [i_min,j_min]  = find(distance==min(distance(:)));
        r0 = [i_min j_min];
        for i = 1:height(distance)
            for j = 1:length(distance)
                r = [i j];
                distance_r0 = vecnorm(r-r0);
                if distance_r0 < 3*sigma
                    h = Neighbourhood(r,r0,sigma);
                    dW = DeltaW(eta,h,X,W(i,j,:));
                    W(i,j,:) = W(i,j,:) + dW;
                end
            end
        end
    end
end


%Initialize for output
termsInit = {zeros(40) zeros(40) zeros(40) zeros(40)};
termsFinal = {zeros(40) zeros(40) zeros(40) zeros(40)};
initWinning_i = zeros(1,150);
initWinning_j = zeros(1,150);
finalWinning_i = zeros(1,150);
finalWinning_j = zeros(1,150);

% Get the output
for i = 1:nInputs
        input = data(i,:);
    for n = 1:length(input)
        termsInit{n} = (W_init(:,:,n) - input(n)).^2;
        termsFinal{n} = (W(:,:,n) - input(n)).^2;
    end
        noise = normrnd(0, 0.02);
        noise2 = normrnd(0, 0.02);
        distanceInit = sqrt(termsInit{1} + termsInit{2} + termsInit{3} + termsInit{4});
        [i_minI,j_minI]  = find(distanceInit==min(distanceInit(:)));
        initWinning_i(i) = i_minI + noise;
        initWinning_j(i) = j_minI + noise2;

        distanceFinal = sqrt(termsFinal{1} + termsFinal{2} + termsFinal{3} ...
            + termsFinal{4});
        [i_minF,j_minF]  = find(distanceFinal==min(distanceFinal(:)));
        finalWinning_i(i) = i_minF + noise;
        finalWinning_j(i) = j_minF + noise2;
end

% Plot
subplot(2,1,1)
hold on
scatter(initWinning_i(1:50),initWinning_j(1:50),40,'red','filled','o')
scatter(initWinning_i(51:100),initWinning_j(51:100),40,'g','filled','o')
scatter(initWinning_i(101:150),initWinning_j(101:150),40,'b','filled','o')
legend('Iris Setosa','Iris Versicolour','Iris Virginica')
title('Winning neuron positions with initial weights')

subplot(2,1,2)
hold on
scatter(finalWinning_i(1:50),finalWinning_j(1:50),40,'r','filled','o')
scatter(finalWinning_i(51:100),finalWinning_j(51:100),40,'g','filled','o')
scatter(finalWinning_i(101:150),finalWinning_j(101:150),40,'b','filled','o')
legend('Iris Setosa','Iris Versicolour','Iris Virginica')
title('Winning neuron positions with trained weights')
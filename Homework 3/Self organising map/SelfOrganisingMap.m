% Maximilian Salén
% 19970105-1576
% Last updated: 2022-10-14


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


for epoch = 1:nEpochs
    eta = eta * exp(-d_eta*epoch);
    sigma = sigma * exp(-d_sigma*epoch);
    for i = 1:nInputs
        randomIndex = randi(nInputs);
        X = data(randomIndex,:);
        distance = norm(W - X)


    end
end
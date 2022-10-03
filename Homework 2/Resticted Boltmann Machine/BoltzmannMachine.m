% Boltzmann machine written by Axel Qvarnström
clear all
close all
clc

% Parameters
N = 3;
MValues = [1, 2, 4, 8];
trials = 100;
miniBatches = 20;
k = 200;
eta = 0.01;
nOut = 300;
nIn = 200;


% Plotting
dKL = zeros(1, length(MValues));
for i = 1:length(MValues)
    M = MValues(i);
    if M < 2^(N-1)-1
        dKL(i) = N - log2(M+1) - ((M+1)/(2^(log2(M + 1))));
    else
        dKL(i) = 0;
    end
end
plot(MValues, dKL)
hold on


% Inputs
XORInputs = [-1,-1,-1; 1,-1,1; -1,1,1; 1,1,-1]';
% allInputs = dec2bin(0:2^N-1)' - '0';
% allInputs(allInputs == 0) = -1;
allInputs = [-1,-1,-1; 1,-1,1; -1,1,1; 1,1,-1; 1,1,1; 1,-1,-1; -1,-1,1; -1,1,-1]';

nPatterns = length(allInputs);
pData = [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0];

% Initializing neurons
v = zeros(N,1);
h = zeros(M,1);

% Initial weights and thresholds
weights = normrnd(0, 1, [M,N]);
for i = 1:size(weights,1) 
    for j = 1:size(weights,2)
        if j == i
            weights(i,i) = 0;      % making the diagonal weights to zero
        end
    end
end

% Initialize thresholds
thetaHidden = 0;
thetaVisible = 0;
for nrOfHiddenNeurons = 1:length(MValues)
    M = MValues(nrOfHiddenNeurons);
    for itrial = 1:trials
    %     deltaW = zeros(M,N);
    %     deltaThetaH = zeros(M,1);
    %     deltaThetaV = zeros(N,1);
%         deltaWeights = 0;
%         deltaThetaH = 0;
%         deltaThetaV = 0;
    
        for iMiniBatch = 1: miniBatches
            % Pick one pattern randomly from x1-x4
            randPatternIndex = randi(nPatterns/2);
            feedPattern = XORInputs(:,randPatternIndex);
            % Initiaize visible neurons as the feed pattern
            v0 = feedPattern;
            
    
            % Update hidden neurons, 
            b0H = weights * v0 - thetaHidden;        % Local field, hidden neurons
            for i = 1:M
                pB0 = probability(b0H(i));
                randomNr = rand;
                if randomNr < pB0
                    h(i) = 1;
                else
                    h(i) = -1;
                end
            end
            
            
            for t = 1:k
                % Update visible neurons
                bV = weights' * h - thetaVisible;
                for j = 1:N
                    pBV = probability(bV(j));
                    randomNr = rand;
                    if randomNr < pBV
                        v(j) = 1;
                    else
                        v(j) = -1;
                    end
                end
    
                % Update hidden neurons
                bH = weights * v - thetaHidden;
                for i = 1:M
                    pBH = probability(bH(i));
                    randomNr = rand;
                    if randomNr < pBH
                        h(i) = 1;
                    else
                        h(i) = -1;
                    end
                end
            end
    
            % Compute weight and threshold increments
            deltaWeights = eta * (tanh(b0H) * v0' - tanh(bH)*v');
            deltaThetaV = -eta*(v0 - v);
            deltaThetaH = -eta*(tanh(b0H) - tanh(bH));
    
            % Updating weight and threshold
            weights = weights + deltaWeights;
            thetaVisible = thetaVisible + deltaThetaV;
            thetaHidden = thetaHidden + deltaThetaH;
        end
    end
    
    pB = zeros(nPatterns,1);
    for iOuter = 1:nOut
        randomPatternIndex = randi(nPatterns);
        feedPattern = allInputs(:,randomPatternIndex);
        % Initiaize visible neurons as the feed pattern
        v = feedPattern;
        
        bH =  weights * v - thetaHidden;
        for i = 1:M
            pBH = probability(bH(i));
            randomNr = rand;
            if randomNr < pBH
                h(i) = 1;
            else
                h(i) = -1;
            end
        end
    
    
        for iInner = 1:nIn
            bV = weights' * h - thetaVisible;
            for j = 1:N
                pBV = probability(bV(j));
                randomNr = rand;
                if randomNr < pBV
                    v(j) = 1;
                else
                    v(j) = -1;
                end
            end
    
            % Update hidden neurons
            bH = weights * v - thetaHidden;
            for i = 1:M
                pBH = probability(bH(i));
                randomNr = rand;
                if randomNr < pBH
                    h(i) = 1;
                else
                    h(i) = -1;
                end
            end
            for iPattern = 1:nPatterns
                if v == allInputs(:,iPattern)
                    pB(iPattern) = pB(iPattern) + 1/(nIn * nOut);
                end
            end
        end
    end
    
    
    
    
    % Calculating the 
    dKLSum = 0;
    for mu = 1:nPatterns
        if pData(mu) ~= 0
            dKLSum = dKLSum + pData(mu) * (log(pData(mu)) - log(pB(mu)));
        end
    end
    
    plot(M,dKLSum,'ro')
end


















            









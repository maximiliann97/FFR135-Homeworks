%Maximilian Salén
%19970105-1576
%Last Updated: 2022-10-03
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
nPatterns = height(x);

% Initialize variables
N = 3;
nHiddenNeurons = [1 2 4 8];
eta = 0.005;
Pdata = [1/4 1/4 1/4 1/4 0 0 0 0];
DKL_sum = zeros(1,length(nHiddenNeurons));
ubDKL = zeros(1,length(nHiddenNeurons));

% Iteration variables
nTrials = 500;
minibatchSize = 40;
k = 500;
N_out = 3000;
N_in = 2000;
numberOfRuns = 3;

for run = 1:numberOfRuns  
    index = 1;
    for M = nHiddenNeurons
        % Initialization dependent on number of neurons
        V = zeros(N,1);
        h = zeros(M,1);
        theta_v = zeros(N,1);
        theta_h = zeros(M,1);
        w = normrnd(0,1, [N M]);
    
    
        for trial = 1:nTrials
            dW = zeros(M,N);
            dTheta_h = zeros(M,1);
            dTheta_v = zeros(N,1);
            for minibatch = 1:minibatchSize
                % Select random pattern x_i, i = 1,2,3,4
                mu = randi(4);
                pattern = x(mu,:);
                v0 = pattern;
    
                b0_h = w'*v0'-theta_h;
                h = Stochastic(b0_h,M);
                
                for t = 1:k %CD_k loop
                    b_v = w*h' - theta_v;
                    V = Stochastic(b_v,N);
                    
                    b_h = w'*V' - theta_h;
                    h = Stochastic(b_h,M);
                end
    
               %Error computation and learning rule
               dW = dW + eta*(tanh(b0_h)*v0 - tanh(b_h)*V);
               dTheta_h = dTheta_h - eta*(tanh(b0_h) - tanh(b_h));
               dTheta_v = dTheta_v - eta*(v0-V)';
    
            end
            
            %Update weights and thresholds
            w = w + dW';
            theta_h = theta_h + dTheta_h;
            theta_v = theta_v + dTheta_v;
    
        end
        
        P_B = zeros(1,nPatterns);
        for i = 1:N_out
            indexPattern = randi(nPatterns);
            V = x(indexPattern,:);
            b_h = w'*V' - theta_h;
            h = Stochastic(b_h,M);
    
         
            for p = 1:N_in
                b_v = w*h' - theta_v;
                V = Stochastic(b_v,N);
        
                b_h = w'*V' - theta_h;
                h = Stochastic(b_h,M);
    
                for j = 1:nPatterns
                    if isequal(V,x(j,:))
                        P_B(j) = P_B(j) + 1/(N_in*N_out);
                    end
                end
            end
        end 
       
    
        DKL = ComputeDKL(Pdata,P_B);
        DKL_sum(index) = DKL_sum(index) + DKL;
        ubDKL(index) = UpperBoundDKL(M,N);
        index = index + 1;       
    end
    fprintf('Run %d\n',run)
end
averageDKL = DKL_sum/numberOfRuns;
% Plot
hold on
plot(nHiddenNeurons, averageDKL, '-o')
plot(nHiddenNeurons, ubDKL)
title('Kullback-Leibler divergence as a function of number of hidden neurons M')
xlabel('M')
ylabel('D_{KL}')
legend('Esitmated D_{KL}', 'Upper bound D_{KL}')






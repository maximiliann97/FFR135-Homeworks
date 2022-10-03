function neuronState = Stochastic(localField,numberOfNeurons)
    P = 1./ (1+exp(-2.*localField));
    neuronState = zeros(1,numberOfNeurons);  

    for i= 1:numberOfNeurons
        r = rand();
        if r < P(i)
            neuronState(i) = 1;
        else
            neuronState(i) = -1;
        end
    end

end


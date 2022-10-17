function W = UpdateCloseNeurons(distance,eta,X,W,r0,sigma)
    
    closeDistances = distance(distance<3*sigma);
    nDist = length(closeDistances);

    for iDist = 1:nDist
        [i,j] = find(distance == closeDistances(iDist));
        r = [i,j];
        W_update = squeeze(W(i,j,:))';
%         h = Neighbourhood(r,r0,sigma);
        h = NeighbourhoodFun(r, r0, sigma);
        dW = DeltaW(eta,h,X,W_update);
        W_update = W_update + dW;
        W(i,j,:) = W_update;
    end

end


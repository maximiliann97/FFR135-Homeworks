function dW = DeltaW(eta,h,X,W)
    dW = eta * h * (X-W);
end


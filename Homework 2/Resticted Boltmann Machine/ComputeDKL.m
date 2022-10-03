function DKL = ComputeDKL(Pdata, P_B)
    DKL = 0;
    for i = 1:4
        if Pdata(i) ~= 0
            DKL = DKL + Pdata(i)*log(Pdata(i)/P_B(i));
        end
    end
end


function ubDKL = UpperBoundDKL(M,N)
      if (M < 2^(N-1) - 1 )
          ubDKL = N - log2(M+1) - (M+1)/2^(log2(M+1));
      else
          ubDKL = 0;
      end
end


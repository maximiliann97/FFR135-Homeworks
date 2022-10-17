function h = Neighbourhood(r, r0, sigma)
    nom = norm(r-r0)^2;
    denom = 2*sigma^2;
    h = exp(-nom / denom);
end


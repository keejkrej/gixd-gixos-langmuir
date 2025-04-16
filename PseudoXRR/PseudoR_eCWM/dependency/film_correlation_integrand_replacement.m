function result = film_correlation_integrand_replacement(r, qxy, eta, zeta, amin)
% replacement integrand: r^(1-eta) * (exp(-eta*besselk)-1) * besselj
% reason: the integration r^(1-eta)*besslej would be divergent under
% numerical integration in matlab. Therefore the analytical result of the
% integral of this part should be seperated from the integrand, such that
% the rest can be integrated in matlab.
    result = sqrt(r.^2+amin^2).^(1-eta) .*(exp(-eta*besselk(0,sqrt(r.^2+amin^2)/zeta))-1).*besselj(0,sqrt(r.^2+amin^2)*qxy);
end


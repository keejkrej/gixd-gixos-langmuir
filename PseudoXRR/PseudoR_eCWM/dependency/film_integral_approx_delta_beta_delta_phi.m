function result = film_integral_approx_delta_beta_delta_phi(beta, phi, kbT_gamma, wave_number, alpha, zeta, amin)
% for fluctuating thin film
% integration over beta resolution and phi resolution at a given beta at a given phi
%   qxy(beta, phi)
% using Daillant 2009 approximation form
gamma_E = double(eulergamma);
qxy = wave_number * sqrt((cosd(beta).*sind(phi)).^2+(cosd(alpha)-cosd(beta).*cosd(phi)).^2);
qz = wave_number*(sind(alpha)+sind(beta));
eta = kbT_gamma/2/pi* qz.^2;
% using kbT_gamma form for eta<1.5 

if zeta > amin/pi
    result = kbT_gamma.* zeta.^eta .* qxy.^eta ./ (qxy.^2 + zeta^2*qxy.^4);
else
    result = kbT_gamma.* (amin/pi).^eta .* qxy.^(eta-2) ;

end


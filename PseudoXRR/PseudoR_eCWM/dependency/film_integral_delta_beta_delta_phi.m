function result = film_integral_delta_beta_delta_phi(beta, phi, kbT_gamma, wave_number, alpha, zeta, amin)
% for fluctuating thin film
% integration over beta resolution and phi resolution at a given beta at a given phi
%   qxy(beta, phi)
gamma_E = double(eulergamma);
qxy = wave_number * sqrt((cosd(beta).*sind(phi)).^2+(cosd(alpha)-cosd(beta).*cosd(phi)).^2);
qz = wave_number*(sind(alpha)+sind(beta));
eta = kbT_gamma/2/pi* qz.^2;

% beta is n*n matrix so it seems it is necessary to loop
% matlab numerical integration
C_prime = zeros(size(beta,1),size(beta,2));
for idx1  = 1:size(beta,1)
    for idx2 = 1:size(beta,2)
        fun = @(x) film_correlation_integrand_replacement(x, qxy(idx1, idx2), eta(idx1, idx2), zeta, amin);
        C_prime(idx1, idx2) = 2*pi * integral(fun, 0.001, 8*zeta);
    end
end

% r_step = 0.001;
% r = [0.001:r_step:8*round(zeta)];
% for idx = 1:length(r)
%     crosssectionIntegrand_replace(:,:,idx) = r(idx).^(1-eta) .* (exp(-eta.*besselk(0, r(idx)./zeta))-1) .* besselj(0,qxy.*r(idx));
%     C_prime(:,:) = 2*pi* sum(crosssectionIntegrand_replace,3) *r_step; 
% end

% using kbT_gamma form for eta<1.5 
% result = (kbT_gamma .* qxy.^(eta-2) + C_prime./ qz.^2) .* zeta.^eta .* exp(eta*(log(2)-gamma_E));
% use precise expression:
xi = (2.^(1-eta).*gamma(1-0.5*eta)./gamma(0.5*eta)) *2*pi./qz.^2;
result = (xi .* qxy.^(eta-2) + C_prime./ qz.^2) .* zeta.^eta .* exp(eta*(log(2)-gamma_E));

end


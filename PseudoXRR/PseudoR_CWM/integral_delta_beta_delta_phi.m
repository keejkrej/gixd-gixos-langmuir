function result = integral_delta_beta_delta_phi(beta, phi, kbT_gamma, wave_number, alpha, Qmax)
% integration over beta resolution and phi resolution at a given beta at a given phi
%   qxy(beta, phi)
qxy = wave_number * sqrt((cosd(beta).*sind(phi)).^2+(cosd(alpha)-cosd(beta).*cosd(phi)).^2); 
qz = wave_number*(sind(alpha)+sind(beta));
eta = kbT_gamma/2/pi* qz.^2;
result = kbT_gamma .* qxy.^(eta-2) ./ Qmax.^eta;
% use precise expression:
xi = (2.^(1-eta).*gamma(1-0.5*eta)./gamma(0.5*eta)) *2*pi./qz.^2;
%result = xi .* qxy.^(eta-2) ./ Qmax.^eta;
end


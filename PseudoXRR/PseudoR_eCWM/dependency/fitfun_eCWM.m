function error = fitfun_eCWM(qxy0_array, intensity_array, coeff, params)
% function to compute R* along phi at different beta using eCWM, and calculate the
% sqrd error between the computation and data
% 2009 Daillant approximation is used
% energy, alpha, Rqxy_HWHM, DSqxy_HWHM, DSbeta_HWHM, tension, temp, kapa, amin

% coeff(1): kapa
kapa = coeff(1);

% basic
gamma_E = double(eulergamma);
kb = 1.381E-23; % Boltzmann constant, J/K
wavelength = 12.4/params.energy;
wave_number = 2*pi / wavelength;
qz_array = (sind(params.alpha)+sind(params.beta_array)) * wave_number;

% phi opening
phi_array = 2*asind(qxy0_array/(2*wave_number)); % offspecular angle [deg]
phi_HWHM = rad2deg(params.DSqxy_HWHM*wavelength/2/pi); % % HWHM of phi resolution, [deg]
phi_upper = phi_array + phi_HWHM;
phi_lower = phi_array - phi_HWHM;

% qxy
for idx = 1:length(phi_array)
    qxy_array(:,idx) = 2*pi/wavelength * sqrt((cosd(params.beta_array).*sind(phi_array(idx))).^2+(cosd(params.alpha)-cosd(params.beta_array).*cosd(phi_array(idx))).^2);
end

% beta opening
beta_upper = params.beta_array + params.DSbeta_HWHM;
beta_lower = params.beta_array - params.DSbeta_HWHM;

% tension term
kbT_gamma = kb*params.temp/params.tension*10^20; % kbT/gamma, prefactor
zeta = sqrt(abs(kapa)*kb*params.temp/params.tension)*10^10; % [A]
eta = kbT_gamma/2/pi*qz_array.^2;

% diffuse scattering via integral
DS_term = ones(length(params.beta_array),length(phi_array));
for beta_idx = 1:length(params.beta_array)
    for phi_idx = 1:length(phi_array)
        fun = @(tt,tth) film_integral_approx_delta_beta_delta_phi(tt, tth, kbT_gamma, wave_number, params.alpha, zeta, params.amin);
        DS_term(beta_idx,phi_idx) = integral2(fun, beta_lower(beta_idx), beta_upper(beta_idx), phi_lower(phi_idx), phi_upper(phi_idx));
    end
end

% scale intensity with DS_term on the 2nd column
intensity_array_scaled = intensity_array.*(DS_term(:,2)./intensity_array(:,2));

%% compute error on GIXOS_sim
error = sum(sum((log10(intensity_array_scaled) - log10(DS_term)).^2));

end
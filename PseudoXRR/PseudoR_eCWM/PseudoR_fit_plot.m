%% replot the PseudoR, GIXOS with fitted structure factor

clear all;
close all;

%% geometry
gamma_E = double(eulergamma);
geo.Qc =0.0217;
geo.energy=15000; 
geo.alpha_i=0.07;
geo.Ddet=560.7;
geo.pixel = 0.075;
geo.footprint= 57;
geo.wavelength = 12404/geo.energy;
geo.qxy0 = 0.03; % 0.04 for dppe, 0.03 for popc
geo.qxy_bkg = 0.4; % bkg region from the GISAXS data itself
geo.RqxyHW = 0.0002 ; %[A^-1] resolution for XRR
geo.DSresHW = 0.003; % HW of the DS resolution at specular
geo.DSqxyHW = 2*geo.DSresHW; % HW of the region of interest in qxy0 for integration, set to 2* DS resolution (res = 0.003A^-1)

%%%%%%%%%%%%%%%%%%%%%%
%% file
%%%%%%%%%%%%%%%%%%%%%%
% file
path = 'U:\p08\2023\data\11017009\shared\SC_analysis\pseudoXRR2\';
sample = 'popc';
scan = 290;

path_sld = 'U:\p08\2023\data\11017009\shared\SC_analysis\fitting\pseudoXRR2\';
file_sld = 'sld_popc_00290_SF.dat';
fit_scaling = 223; % 

bkgsample = 'trough_chamber_bkg';
bkgscan = 319;

kb = 1.381E-23; % Boltzmann constant, J/K
tension = 38E-3; % tension, [N/m]
kapa = 5;  % [kT]
temperature = 293; % [K]
zeta = sqrt(kapa*kb*temperature/tension)*10^10; % [A]
amin = 5; % minimal wavelength cutoff of the surface CW
incorrect_roughness = 1.9 ; % [A] roughness incorrectly determined from DS without considering DS/RRF
RFscaling = fit_scaling; % for both
%%
refl_file = strcat(sample,'_',num2str(scan,'%05d'),'_R.dat');
refl = importdata(strcat(path,refl_file));
SF_file = strcat(sample,'_',num2str(scan,'%05d'),'_SF.dat');
SF = importdata(strcat(path, SF_file));
DS2RRF_file = strcat(sample,'_',num2str(scan,'%05d'),'_DS2RRF.dat');
DS2RRF = importdata(strcat(path, DS2RRF_file));
% load sld from fitting to reconstruct SF
fit_sld = load(strcat(path_sld,file_sld));

%%
% arrange data structure
GIXOS.Qz = refl.data(:,1);

%% create pseudo reflectivity R = GIXOS/(DS/RRF)*RF/transmission
GIXOS.fresnel = GIXOS_fresnel(GIXOS.Qz(:,1),geo.Qc);
GIXOS.transmission = GIXOS_Tsqr(GIXOS.Qz(:,1),geo.Qc, geo.energy, geo.alpha_i, geo.Ddet, geo.footprint);
GIXOS.DS_RRF = DS2RRF.data(:,2);
GIXOS.RRF_term = SF.data(:,6);
% pseudo reflecitivity Qz, R, dR, dQ
GIXOS.refl = refl.data;
roughness = SF.data(:,5);
%% structure factor reconstructed from sld
clear fit_drho_dz;
fit_drho_dz = drho_dz(fit_sld); % drho_dz array: z, sld, drho/dz
fit_data = RRF_DWBA(GIXOS.Qz, fit_drho_dz);
fit_data.SF = fit_data.RRF*fit_scaling;

%% plot
close(findobj('name','refl'));
op1 = figure('name','refl','Position',[100,100,600,500]);
pa1 = axes('Parent',op1);
hold on
plot(GIXOS.refl(:,1), GIXOS.refl(:,2).*GIXOS.DS_RRF/GIXOS.DS_RRF(1), 'ko' ,'MarkerSize', 3, 'DisplayName', 'without DS/RRF correction'); % without DS/(R/RF) correction
plot(GIXOS.fresnel(:,1), RFscaling*GIXOS.fresnel(:,2).*exp(-(incorrect_roughness*GIXOS.refl(:,1)).^2), '--k', 'LineWidth', 1.5, 'DisplayName', ['Fresnel: \sigma=', num2str(incorrect_roughness, '%.1f'),'A']);
plot(GIXOS.refl(:,1), GIXOS.refl(:,2).*GIXOS.transmission(:,4), 'ro' ,'MarkerSize', 3, 'DisplayName', 'without Tr correction'); % without transmission correction
errorbar(GIXOS.refl(:,1), GIXOS.refl(:,2), GIXOS.refl(:,3), 'bo' ,'MarkerSize', 3, 'DisplayName', 'with Tr correction'); % with transmission correction
plot(GIXOS.fresnel(:,1), GIXOS.fresnel(:,2).*fit_data.SF.*GIXOS.RRF_term , '-b', 'LineWidth', 1.5, 'DisplayName', ['|\Phi|^2 with \sigma=', num2str(roughness(1), '%.2f'),'A to ', num2str(roughness(end), '%.2f'),'A'])
hold off
xlim([0 1.05])
xlabel(['Q_z [' char(197) '^-^1]'],'FontSize',14);
ylabel('I [a.u.]','FontSize',14);
pa1.FontSize = 14;
pa1.LineWidth = 1;
pa1.XTick = 0:0.2:1;
pa1.TickDir = 'out';
set(gca, 'YScale', 'Log');
legend('location','NorthEast','box','off');
saveas(op1, strcat(path,sample,'_',num2str(scan,'%05d'),'_Rfit.jpg'));

%%
close(findobj('name','RRF'));
op2 = figure('name','RRF','Position',[100,100,600,500]);
pa2 = axes('Parent',op2);
hold on
plot(GIXOS.refl(:,1), GIXOS.refl(:,2)./GIXOS.fresnel(:,2).*GIXOS.DS_RRF/GIXOS.DS_RRF(1), 'ko' ,'MarkerSize', 3, 'DisplayName', 'without DS/RRF correction'); % without DS/(R/RF) correction
plot(GIXOS.fresnel(:,1), RFscaling.*exp(-(incorrect_roughness*GIXOS.refl(:,1)).^2), '--k', 'LineWidth', 1.5, 'DisplayName', ['Fresnel: \sigma=', num2str(incorrect_roughness, '%.1f'),'A']);
plot(GIXOS.refl(:,1), GIXOS.refl(:,2).*GIXOS.transmission(:,4)./GIXOS.fresnel(:,2), 'ro' ,'MarkerSize', 3, 'DisplayName', 'without Tr correction'); % without transmission correction
errorbar(GIXOS.refl(:,1), GIXOS.refl(:,2)./GIXOS.fresnel(:,2), GIXOS.refl(:,3)./GIXOS.fresnel(:,2), 'bo' ,'MarkerSize', 3, 'DisplayName', 'with Tr correction'); % with transmission correction
plot(GIXOS.fresnel(:,1), fit_data.SF.*GIXOS.RRF_term, '-b', 'LineWidth', 1.5, 'DisplayName', ['|\Phi|^2 with \sigma=', num2str(roughness(1), '%.2f'),'A to ', num2str(roughness(end), '%.2f'),'A'])
hold off
xlim([0 1.05])
xlabel(['Q_z [' char(197) '^-^1]'],'FontSize',14);
ylabel('R/R_F [a.u.]','FontSize',14);
pa2.FontSize = 14;
pa2.LineWidth = 1;
pa2.XTick = 0:0.2:1;
pa2.TickDir = 'out';
set(gca, 'YScale', 'Log');
legend('location','SouthWest','box','off');
saveas(op2, strcat(path,sample,'_',num2str(scan,'%05d'),'_RRFfit.jpg'));
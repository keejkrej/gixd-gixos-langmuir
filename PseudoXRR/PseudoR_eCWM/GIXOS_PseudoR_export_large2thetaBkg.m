%% GIXOS reduction macro for thin film %%
% load and export the 1D GIXOS data as
% - I(Qz)
% - create its pseudo reflectivity 
% R = GIXOS*RF/transmission*factor, Qz, R, dR, dQ
% Chen 06.06.2023
% include absolute scaling and bkg subtraction
% fit large 2theta bkg

clear all;
close all;

%%
colorset = [    0 0.4470 0.7410; ...
                0.8500 0.3250 0.0980; ...
                0.9290 0.6940 0.1250; ...
                0.4940 0.1840 0.5560; ...
                0.4660 0.6740 0.1880; ...
                0.3010 0.7450 0.9330; ...
                0.6350 0.0780 0.1840; ...
                0 0.4470 0.7410; ...
                0.8500 0.3250 0.0980; ...
                0.9290 0.6940 0.1250; ...
    ];

%% geometry
gamma_E = double(eulergamma);
geo.Qc =0.0216;
geo.energy=15000; 
geo.alpha_i=0.07;
geo.Ddet=560.7;
geo.pixel = 0.075;
geo.footprint= 30;
geo.wavelength = 12404/geo.energy;
geo.qxy0 = 0.03; % 0.04 for dppe, 0.03 for popc
geo.qxy_bkg = [0.35:0.01:0.45]; % bkg region from the GISAXS data itself
geo.RqxyHW = 0.0002 ; %[A^-1] resolution for XRR
geo.DSresHW = 0.003; % HW of the DS resolution at specular
geo.DSqxyHW = 2*geo.DSresHW; % HW of the region of interest in qxy0 for integration, set to 2* DS resolution (res = 0.003A^-1)
geo.qz_selected = [0.1:0.1:0.9]';
%%%%%%%%%%%%%%%%%%%%%%
%% file
%%%%%%%%%%%%%%%%%%%%%%
% file
path = '/gpfs/current/processed/GID/';
path_out = '/gpfs/current/processed/PseudoXRR/';
sample = 'dopc';
scan =19;
I0_sample_chamber = 1;
bkgsample = 'dopc';
bkgscan = 27;

kb = 1.381E-23; % Boltzmann constant, J/K
tension = (73-0.1)/1000; % tension, [N/m]
kapa = 0.1;  % [kT]
temperature = 293; % [K]
zeta = sqrt(kapa*kb*temperature/tension)*10^10; % [A]
amin = 5; % minimal wavelength cutoff of the surface CW
incorrect_roughness = 1.9 ; % [A] roughness incorrectly determined from DS without considering DS/RRF
RFscaling = 3.5*I0_sample_chamber*10^10*30 *(pi/180)^2 * (9.42E-6)^2 /sind(geo.alpha_i)*4; % for both
%%
% file name of the GISAXS to be imported
fileprefix = strcat('GID_',sample,'_',num2str(scan,'%05d'),'_angle');
Ifilename = strcat(path, fileprefix, '_I.dat');
tthfilename = strcat(path, fileprefix, '_tth.dat');
ttfilename = strcat(path, fileprefix, '_tt.dat');
importdata.Intensity = load(Ifilename);
importdata.tth = load(tthfilename);
importdata.tt = load(ttfilename);

% file name of the bkg GISAXS to be imported
bkgfileprefix = strcat('GID_',bkgsample,'_',num2str(bkgscan,'%05d'),'_angle');
bkgIfilename = strcat(path, bkgfileprefix, '_I.dat');
bkgtthfilename = strcat(path, bkgfileprefix, '_tth.dat');
bkgttfilename = strcat(path, bkgfileprefix, '_tt.dat');
importbkg.Intensityraw = load(bkgIfilename);
importbkg.Intensity = importbkg.Intensityraw*I0_sample_chamber;
importbkg.tth = load(bkgtthfilename);
importbkg.tt = load(bkgttfilename);

%% binning
binsize = 10;
groupnumber = floor(size(importdata.Intensity, 1)/binsize);

for groupidx = 1:groupnumber
    binneddata.Intensity(groupidx,:) = sum(importdata.Intensity((groupidx-1)*binsize+1:groupidx*binsize,:));
    binneddata.tt(groupidx,:) =  mean(importdata.tt((groupidx-1)*binsize+1:groupidx*binsize,:));
    binneddata.tth = importdata.tth;
    
    binnedbkg.Intensity(groupidx,:) = sum(importbkg.Intensity((groupidx-1)*binsize+1:groupidx*binsize,:));
    binnedbkg.tt(groupidx,:) =  mean(importbkg.tt((groupidx-1)*binsize+1:groupidx*binsize,:));
    binnedbkg.tth = importbkg.tth;
end

importdata = binneddata;
importbkg = binnedbkg;

%%
% angle step size:
geo.tth_step = mean(importdata.tth(2:end) - importdata.tth(1:end-1));
geo.tt_step = mean(importdata.tt(2:end) - importdata.tt(1:end-1));
% remove negative tt
tt_start_idx = find(importdata.tt<=0,1,'last');
importdata.Intensity(1:tt_start_idx,:) = [];
importdata.tt(1:tt_start_idx) = [];
importbkg.Intensity(1:tt_start_idx,:) = [];
importbkg.tt(1:tt_start_idx) = [];

%%
tth = asind(geo.qxy0 * geo.wavelength / 4 / pi)*2;   % tth for the qxy0
tth_bkg = asind(geo.qxy_bkg * geo.wavelength / 4 / pi)*2; % tth for the bkg region in GISAXS
tth_idx = find(importdata.tth>tth,1,'first');
for idx = 1:length(tth_bkg)
    tth_bkg_idx(idx) = find(importdata.tth>tth_bkg(idx),1,'first');   % bkg tth in the GISAXS
end
tth_roiHW = rad2deg(geo.DSqxyHW * geo.wavelength / 2 / pi / cosd(tth/2));   % region of interest of the tth
mat_roiHW = floor(tth_roiHW / geo.tth_step);
tth_roiHW_real = mat_roiHW*geo.tth_step;
geo.DSqxyHW_real = deg2rad(tth_roiHW_real)/2*4*pi/geo.wavelength*cosd(tth/2);

%%
% arrange data structure
GIXOS.tt = importdata.tt;
GIXOS.Qz = 2*pi/geo.wavelength*(sind(GIXOS.tt)+sind(geo.alpha_i));
GIXOS.Qxy_px = 2*pi/geo.wavelength *sqrt((cosd(GIXOS.tt)*sind(tth)).^2+(cosd(geo.alpha_i)-cosd(GIXOS.tt)*cosd(tth)).^2);
GIXOS.Q = sqrt(GIXOS.Qz.^2+GIXOS.Qxy_px.^2);

geo.DSbetaHW = mean(GIXOS.tt(2:end) - GIXOS.tt(1:end-1))/2;
GIXOS.GIXOS_raw = sum(importdata.Intensity(:,tth_idx-mat_roiHW:tth_idx+mat_roiHW),2);
GIXOS.GIXOS_bkg = sum(importbkg.Intensity(:,tth_idx-mat_roiHW:tth_idx+mat_roiHW),2);
for idx = 1:length(tth_bkg_idx)
    GIXOS.GIXOS_largetth_Qz((idx-1)*(length(GIXOS.Qz)-25)+1:idx*(length(GIXOS.Qz)-25),1)=GIXOS.Qz(26:end);
    GIXOS.GIXOS_largetth_Qxy((idx-1)*(length(GIXOS.Qz)-25)+1:idx*(length(GIXOS.Qz)-25),1) = 2*pi/geo.wavelength*sqrt((cosd(GIXOS.tt(26:end)).*sind(tth_bkg(idx))).^2+(cosd(GIXOS.tt(26:end)).*cosd(tth_bkg(idx))-cosd(geo.alpha_i)).^2);
    GIXOS.GIXOS_raw_largetth((idx-1)*(length(GIXOS.Qz)-25)+1:idx*(length(GIXOS.Qz)-25),1) = sum(importdata.Intensity(26:end,tth_bkg_idx(idx)-mat_roiHW:tth_bkg_idx(idx)+mat_roiHW),2);
    GIXOS.GIXOS_bkg_largetth((idx-1)*(length(GIXOS.Qz)-25)+1:idx*(length(GIXOS.Qz)-25),1) = sum(importbkg.Intensity(26:end,tth_bkg_idx(idx)-mat_roiHW:tth_bkg_idx(idx)+mat_roiHW),2);
end
GIXOS.GIXOS_largetth_Q = sqrt(GIXOS.GIXOS_largetth_Qz.^2+GIXOS.GIXOS_largetth_Qxy.^2);
%%
[GIXOS.GIXOS_largetth_Q ,sortIdx] = sort(GIXOS.GIXOS_largetth_Q ,'ascend');
GIXOS.GIXOS_largetth_Qz = GIXOS.GIXOS_largetth_Qz(sortIdx);
GIXOS.GIXOS_largetth_Qxy = GIXOS.GIXOS_largetth_Qxy(sortIdx);
GIXOS.GIXOS_raw_largetth = GIXOS.GIXOS_raw_largetth(sortIdx);
GIXOS.GIXOS_bkg_largetth = GIXOS.GIXOS_bkg_largetth(sortIdx);
GIXOS.bulk_bkg = GIXOS.GIXOS_raw_largetth - GIXOS.GIXOS_bkg_largetth;

bulkfittype = fittype("a+b*exp(x*c)",dependent="y", independent="x", coefficients = ["a" "b" "c"]);
f_baseline = fit(GIXOS.GIXOS_largetth_Q, GIXOS.bulk_bkg ,bulkfittype,'Upper',[mean(GIXOS.bulk_bkg(1:10),1)*5, 1000,10], 'Lower', [0, 0, 0], 'StartPoint', [mean(GIXOS.bulk_bkg(1:10),1), 100, 1]);
GIXOS.bulk_bkg_coeff = coeffvalues(f_baseline);
GIXOS.bulk_bkg_fit = exp(GIXOS.GIXOS_largetth_Q*GIXOS.bulk_bkg_coeff(3))*GIXOS.bulk_bkg_coeff(2)+GIXOS.bulk_bkg_coeff(1);

close(findobj('name','bulkbkg'));
op_bulkbkg = figure('name','bulkbkg','Position',[50,50,300,300]);
hold on
plot(GIXOS.GIXOS_largetth_Q, GIXOS.bulk_bkg, 'ko-');
plot(GIXOS.GIXOS_largetth_Q, GIXOS.bulk_bkg_fit,'r-');
hold off;

%%
% solid angle correspondence correction: the intensity is not corrected for pixel solid angle coverage when rebinned into tt space
GIXOS.fdtt = deg2rad(geo.tt_step) ./ (atan((tand(GIXOS.tt)*geo.Ddet+geo.pixel/2)/geo.Ddet) - atan((tand(GIXOS.tt)*geo.Ddet-geo.pixel/2)/geo.Ddet));   %[rad]
GIXOS.fdtt = GIXOS.fdtt / GIXOS.fdtt(1);

GIXOS.bulk_bkg_apply = (GIXOS.bulk_bkg_coeff(1) + GIXOS.bulk_bkg_coeff(2)*exp(GIXOS.Q*GIXOS.bulk_bkg_coeff(3)));

GIXOS.GIXOS= ((GIXOS.GIXOS_raw -GIXOS.GIXOS_bkg)- GIXOS.bulk_bkg_apply).*GIXOS.fdtt;
GIXOS.error= sqrt(sqrt(abs(GIXOS.GIXOS_raw)).^2 + sqrt(abs(GIXOS.GIXOS_bkg)).^2) .*GIXOS.fdtt;
GIXOS.bkg = 0;
%% GIXOS
close(findobj('name','raw'));
fig_refl=figure('name','raw');
plot(GIXOS.Qz, GIXOS.GIXOS,'ko','MarkerSize',3,'DisplayName','corrected data by chamber transmission bkg');
hold on
plot(GIXOS.Qz, GIXOS.GIXOS_raw,'r-','LineWidth',1.5, 'DisplayName', 'raw data');
plot(GIXOS.Qz, GIXOS.GIXOS_bkg,'b-','LineWidth',1.5, 'DisplayName', 'bkg data at same tth');
plot(GIXOS.Qz, GIXOS.bulk_bkg_apply,'r:','LineWidth',1.5, 'DisplayName', 'raw data baseline extrapolated');
yline(0,'k-.', 'LineWidth', 1.5, 'DisplayName', '0-line');
hold off
%ylim([-6 1])
xlim([0 1.05]);
ax=gca;
xlabel(['Q_z [' char(197) '^-^1]'],'FontSize',12);
ylabel('GIXOS','FontSize',12);
ax.XTick=0:0.2:1;
ax.FontSize = 12;
ax.LineWidth = 2;
ax.TickDir = 'out';
legend('location','NorthEast','box','off');
saveas(fig_refl, strcat(path_out,sample,'_',num2str(scan,'%05d'),'_GIXOS.jpg'));
%% create pseudo reflectivity R = GIXOS/(DS/RRF)*RF/transmission
% select the range of Qz for eta<2
row_etalim = find(GIXOS.Qz<sqrt(4*pi*tension/kb/temperature/10^20),1,'last');
% continue
GIXOS.fresnel = GIXOS_fresnel(GIXOS.Qz(1:row_etalim,1),geo.Qc);
GIXOS.transmission = GIXOS_Tsqr(GIXOS.Qz(1:row_etalim,1),geo.Qc, geo.energy, geo.alpha_i, geo.Ddet, geo.footprint);
GIXOS.dQz = GIXOS_dQz(GIXOS.Qz(1:row_etalim,1),geo.energy, geo.alpha_i, geo.Ddet, geo.footprint);
% integration over beta and phi properly
[GIXOS.DS_RRF, GIXOS.DS_term, GIXOS.RRF_term] = calc_film_DS_RRF_integ(GIXOS.tt(1:row_etalim),geo.qxy0,geo.energy/1000, geo.alpha_i, geo.RqxyHW, geo.DSqxyHW_real, geo.DSbetaHW, tension, temperature, kapa, amin);
% pseudo reflecitivity Qz, R, dR, dQ
GIXOS.refl = [GIXOS.Qz(1:row_etalim) (GIXOS.GIXOS(1:row_etalim,:)-GIXOS.bkg)./GIXOS.DS_RRF.*GIXOS.fresnel(:,2)./GIXOS.transmission(:,4) GIXOS.error(1:row_etalim,:)./GIXOS.DS_RRF.*GIXOS.fresnel(:,2)./GIXOS.transmission(:,4) GIXOS.dQz(:,5)];
%% structure factors
r_step = 0.001;
r = sqrt([0.001:r_step:8*round(zeta)].^2+amin^2); % include molecular cutoff
eta = kb*temperature/tension* 10^20 /2/pi*(GIXOS.Qz(1:row_etalim)).^2;
for idx = 1:row_etalim
    C_integrand(idx,:) = 2*pi* r.^(1-eta(idx)).*(exp(-eta(idx).*besselk(0,r/zeta))-1); % eta and zeta dependent factor C(eta, zeta)
end
C = sum(C_integrand,2) * r_step;
roughness = sqrt(kb*temperature/tension*10^20/2/pi*(log(1/geo.RqxyHW/zeta)+gamma_E - log(2)) - log(1+(geo.RqxyHW).^(2-eta)/4/pi .*C) ./(GIXOS.Qz(1:row_etalim)).^2); % kapa-CW roughness [A]
clear r_step r eta;

% structure factor after extracting out the kapa / CW roughness term and
% pre-factors: Qz, SF, dSF, dQz, roughness and exp(-q2sigma2) under refl resolution
GIXOS.SF = [GIXOS.Qz(1:row_etalim) (GIXOS.GIXOS(1:row_etalim,:)-GIXOS.bkg)./GIXOS.DS_term./GIXOS.transmission(:,4) GIXOS.error(1:row_etalim,:)./GIXOS.DS_term./GIXOS.transmission(:,4) GIXOS.dQz(:,5) roughness exp(-(GIXOS.Qz(1:row_etalim).*roughness).^2)];

%% output pseudo-reflectivity
xrrfilename = strcat(path_out,sample,'_',num2str(scan,'%05d'),'_R.dat');
xrrfid = fopen(xrrfilename,'w');
fprintf(xrrfid,...
    '# files\nsample file: %s\nbackground file: %s\nbkg by fitting Q dependence for a sum of GIXOS at qxy_0 between %f and %f (parameters: %.3f + %.3f * exp(%.3f*Q))\n',...
    fileprefix, bkgfileprefix, geo.qxy_bkg(1), geo.qxy_bkg(end),GIXOS.bulk_bkg_coeff(1), GIXOS.bulk_bkg_coeff(2) ,GIXOS.bulk_bkg_coeff(3));
fprintf(xrrfid,...   
    '# geometry\nenergy [eV]: %.2f\nincidence [deg]: %f\nfootprint [mm]: %.1f\nsdd [mm]: %.2f\nqxy resolution HWHM at specular [A^-1]: %f\nphi_step [deg]: %f\nbeta_step [deg]: %f\n',...
    geo.energy, geo.alpha_i, geo.footprint, geo.Ddet, geo.DSresHW, geo.tth_step, geo.tt_step);
fprintf(xrrfid,...   
    '# DS-XRR conversion optics setting\nphi [deg]: %f\nqxy(beta=0) [A^-1]: %f\nphi integration HW [deg]: %f\ncorresponding qxy HW [A^-1]: %f\nR qxy resolution setting HWHM [A^-1]: %f\nscaling: %f\n',...
    tth, geo.qxy0, tth_roiHW_real, geo.DSqxyHW_real, geo.RqxyHW, RFscaling);
fprintf(xrrfid,...  
    '# DS-XRR conversion sample setting\ntension [N/m]: %f\ntemperature [K]: %.1f\nk_c [kbT]: %.1f\nCW short cutoff [A]: %f\nCW and Kapa roughness [A]: %f to %f\n',...
    tension, temperature, kapa, amin, roughness(1), roughness(end));
fprintf(xrrfid,... 
    '# data\nqz\tR\tdR\tdqz\n[A^-1]\t[a.u.]\t[a.u.]\t[A^-1]\n');
dlmwrite(xrrfilename,GIXOS.refl,'delimiter','\t','-append')  % pseudo reflectivity Qz, R, dR, dQ
fclose(xrrfid);


DS2RRFfilename = strcat(path_out,sample,'_',num2str(scan,'%05d'),'_DS2RRF.dat');
DS2RRFfid = fopen(DS2RRFfilename,'w');
fprintf(DS2RRFfid,...
    '# files\nsample file: %s\nbackground file: %s\nbulk bkg by fitting Q dependence for a sum of GIXOS at qxy_0 between %f and %f (parameters: %.3f + %.3f * exp(%.3f*Q))\n',...
    fileprefix, bkgfileprefix, geo.qxy_bkg(1), geo.qxy_bkg(end),GIXOS.bulk_bkg_coeff(1), GIXOS.bulk_bkg_coeff(2) ,GIXOS.bulk_bkg_coeff(3));
fprintf(DS2RRFfid,...   
    '# geometry\nenergy [eV]: %.2f\nincidence [deg]: %f\nfootprint [mm]: %.1f\nsdd [mm]: %.2f\nqxy resolution HWHM at specular [A^-1]: %f\nphi_step [deg]: %f\nbeta_step [deg]: %f\n',...
    geo.energy, geo.alpha_i, geo.footprint, geo.Ddet, geo.DSresHW, geo.tth_step, geo.tt_step);
fprintf(DS2RRFfid,...   
    '# DS-XRR conversion optics setting\nphi [deg]: %f\nqxy(beta=0) [A^-1]: %f\nphi integration HW [deg]: %f\ncorresponding qxy HW [A^-1]: %f\nR qxy resolution setting HWHM [A^-1]: %f\nscaling: %f\n',...
    tth, geo.qxy0, tth_roiHW_real, geo.DSqxyHW_real, geo.RqxyHW, RFscaling);
fprintf(DS2RRFfid,...  
    '# DS-XRR conversion sample setting\ntension [N/m]: %f\ntemperature [K]: %.1f\nk_c [kbT]: %.1f\nCW short cutoff [A]: %f\nCW and Kapa roughness [A]: %f to %f\n',...
    tension, temperature, kapa, amin, roughness(1), roughness(end));
fprintf(DS2RRFfid,... 
    '# data\nqz\tDS/(R/RF)\n[A^-1]\t[a.u.]\n');
dlmwrite(DS2RRFfilename,[GIXOS.Qz(1:row_etalim), GIXOS.DS_RRF],'delimiter','\t','-append')  % ratio between DS and RRF
fclose(DS2RRFfid);

% structure factor
SFfilename = strcat(path_out,sample,'_',num2str(scan,'%05d'),'_SF.dat');
SFfid = fopen(SFfilename,'w');
fprintf(SFfid,...
    '# pure structure factor and kapa/cw roughness with its decay term under given XRR resolution\n# files\nsample file: %s\nbackground file: %s\nbulk bkg by fitting Q dependence for a sum of GIXOS at qxy_0 between %f and %f (parameters: %.3f + %.3f * exp(%.3f*Q))\n',...
    fileprefix, bkgfileprefix, geo.qxy_bkg(1), geo.qxy_bkg(end),GIXOS.bulk_bkg_coeff(1), GIXOS.bulk_bkg_coeff(2) ,GIXOS.bulk_bkg_coeff(3));
fprintf(SFfid,...   
    '# geometry\nenergy [eV]: %.2f\nincidence [deg]: %f\nfootprint [mm]: %.1f\nsdd [mm]: %.2f\nqxy resolution HWHM at specular [A^-1]: %f\nphi_step [deg]: %f\nbeta_step [deg]: %f\n',...
    geo.energy, geo.alpha_i, geo.footprint, geo.Ddet, geo.DSresHW, geo.tth_step, geo.tt_step);
fprintf(SFfid,...   
    '# DS-XRR conversion optics setting\nphi [deg]: %f\nqxy(beta=0) [A^-1]: %f\nphi integration HW [deg]: %f\ncorresponding qxy HW [A^-1]: %f\nR qxy resolution setting HWHM [A^-1]: %f\nscaling: %f\n',...
    tth, geo.qxy0, tth_roiHW_real, geo.DSqxyHW_real, geo.RqxyHW, RFscaling);
fprintf(SFfid,...  
    '# DS-XRR conversion sample setting\ntension [N/m]: %f\ntemperature [K]: %.1f\nk_c [kbT]: %.1f\nCW short cutoff [A]: %f\nCW and Kapa roughness [A]: %f to %f\n',...
    tension, temperature, kapa, amin, roughness(1), roughness(end));
fprintf(SFfid,... 
    '# data\nqz\tSF\tdSF\tdQz\tsigma_R\texp(-qz2sigma2)\n[A^-1]\t[a.u.]\t[a.u.]\t[A^-1]\t[A^-1]\t[a.u.]\n');
dlmwrite(SFfilename,GIXOS.SF,'delimiter','\t','-append')  % structure factor and xrr roughness term
fclose(SFfid);
%% plot
close(findobj('name','refl'));
op1 = figure('name','refl','Position',[100,100,600,500]);
pa1 = axes('Parent',op1);
hold on
plot(GIXOS.refl(:,1), GIXOS.refl(:,2).*GIXOS.DS_RRF/GIXOS.DS_RRF(1), 'ko' ,'MarkerSize', 3, 'DisplayName', 'without DS/RRF correction'); % without DS/(R/RF) correction
plot(GIXOS.fresnel(:,1), RFscaling*GIXOS.fresnel(:,2).*exp(-(incorrect_roughness*GIXOS.refl(:,1)).^2), '--k', 'LineWidth', 1.5, 'DisplayName', ['Fresnel: \sigma=', num2str(incorrect_roughness, '%.1f'),'A']);
plot(GIXOS.refl(:,1), GIXOS.refl(:,2).*GIXOS.transmission(:,4), 'ro' ,'MarkerSize', 3, 'DisplayName', 'without Tr correction'); % without transmission correction
errorbar(GIXOS.refl(:,1), GIXOS.refl(:,2), GIXOS.refl(:,3), 'bo' ,'MarkerSize', 3, 'DisplayName', 'with Tr correction'); % with transmission correction
plot(GIXOS.fresnel(:,1), RFscaling*GIXOS.fresnel(:,2).*exp(-(roughness.*GIXOS.refl(:,1)).^2), '-b', 'LineWidth', 1.5, 'DisplayName', ['Fresnel: \sigma=', num2str(roughness(1), '%.2f'),'A to ', num2str(roughness(end), '%.2f'),'A'])
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
saveas(op1, strcat(path_out,sample,'_',num2str(scan,'%05d'),'_R.jpg'));

%%
close(findobj('name','RRF'));
op2 = figure('name','RRF','Position',[100,100,600,500]);
pa2 = axes('Parent',op2);
hold on
plot(GIXOS.refl(:,1), GIXOS.refl(:,2)./GIXOS.fresnel(:,2).*GIXOS.DS_RRF/GIXOS.DS_RRF(1), 'ko' ,'MarkerSize', 3, 'DisplayName', 'without DS/RRF correction'); % without DS/(R/RF) correction
plot(GIXOS.fresnel(:,1), RFscaling.*exp(-(incorrect_roughness*GIXOS.refl(:,1)).^2), '--k', 'LineWidth', 1.5, 'DisplayName', ['Fresnel: \sigma=', num2str(incorrect_roughness, '%.1f'),'A']);
plot(GIXOS.refl(:,1), GIXOS.refl(:,2).*GIXOS.transmission(:,4)./GIXOS.fresnel(:,2), 'ro' ,'MarkerSize', 3, 'DisplayName', 'without Tr correction'); % without transmission correction
errorbar(GIXOS.refl(:,1), GIXOS.refl(:,2)./GIXOS.fresnel(:,2), GIXOS.refl(:,3)./GIXOS.fresnel(:,2), 'bo' ,'MarkerSize', 3, 'DisplayName', 'with Tr correction'); % with transmission correction
plot(GIXOS.fresnel(:,1), RFscaling.*exp(-(roughness.*GIXOS.refl(:,1)).^2), '-b', 'LineWidth', 1.5, 'DisplayName', ['Fresnel: \sigma=', num2str(roughness(1), '%.2f'),'A to ', num2str(roughness(end), '%.2f'),'A'])
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
saveas(op2, strcat(path_out,sample,'_',num2str(scan,'%05d'),'_RRF.jpg'));
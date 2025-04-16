%% GIXOS reduction macro %%
% load and export the 1D GIXOS data as
% - I(Qz)
% - create its pseudo reflectivity 
% R = GIXOS*RF/transmission*factor, Qz, R, dR, dQ
% Chen 11.06.2023
% include absolute scaling and bkg subtraction
% bkg is from large 2theta

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
geo.qxy0 = [0.005, 0.02, 0.03, 0.04, 0.06]; % selected qxy0 for GIXOS
geo.qxy_bkg = [0.35:0.01:0.45]; % bkg region from the GISAXS data itself
geo.RqxyHW = 2E-4 ; %[A^-1] resolution for XRR, both direction 15px HW on LISA, 15keV, dqy = 5E-6 - 2E-4 (beta dependent), dqx = 6E-3
geo.DSresHW = 0.003; % HW of the DS resolution at specular
geo.DSqxyHW = 2*geo.DSresHW; % HW of the region of interest in qxy0 for integration, set to 2* DS resolution (res = 0.003A^-1)
geo.qz_selected = [0.1:0.1:0.9]';
%%%%%%%%%%%%%%%%%%%%%%
%% file
%%%%%%%%%%%%%%%%%%%%%%
% file
path = '/gpfs/current/processed/GID/';
path_out = '/gpfs/current/processed/PseudoXRR/';
sample = 'water';
scan = 43;
I0_sample_chamber = 1;

bkgsample = 'dopc';
bkgscan = 27;

kb = 1.381E-23; % Boltzmann constant, J/K
tension = 73E-3; % tension, [N/m]
temperature = 293; % [K]
amin = 3.1; % minimal wavelength cutoff of the surface CW
incorrect_roughness = [2.7, 2.4, 2.1, 2, 1.8]; % [A] roughness incorrectly determined from DS without considering DS/RRF
intrinsic_roughness = 0 ; % [A]
% roughness = 3.4; % roughness for Fresnel [A]
roughness = sqrt(kb*temperature/tension*10^20/2/pi*log(pi/amin/geo.RqxyHW)); % CW roughness [A]
effective_roughness = sqrt(roughness^2+intrinsic_roughness^2); % effective roughness for Fresnel [A]
RFscaling = 3.8*I0_sample_chamber*10^10*30*(pi/180)^2 * (9.42E-6)^2 /sind(geo.alpha_i)*4; 

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
for idx = 1:length(tth)
    tth_idx(idx) = find(importdata.tth>tth(idx),1,'first');
end
for idx = 1:length(tth_bkg)
    tth_bkg_idx(idx) = find(importdata.tth>tth_bkg(idx),1,'first');   % bkg tth in the GISAXS
end
tth_roiHW = rad2deg(geo.DSqxyHW * geo.wavelength / 2 / pi ./ cosd(tth/2));   % region of interest of the tth
mat_roiHW = floor(tth_roiHW / geo.tth_step);
tth_roiHW_real = mat_roiHW*geo.tth_step;
geo.DSqxyHW_real = deg2rad(tth_roiHW_real)/2*4*pi/geo.wavelength .*cosd(tth/2);

%%
% arrange data structure
GIXOS.tt = importdata.tt;
GIXOS.Qz = 2*pi/geo.wavelength*(sind(GIXOS.tt)+sind(geo.alpha_i));
GIXOS.Qxy_px = 2*pi/geo.wavelength *sqrt((cosd(GIXOS.tt)*sind(tth)).^2+(cosd(geo.alpha_i)-cosd(GIXOS.tt)*cosd(tth)).^2);
GIXOS.Q = sqrt(GIXOS.Qz.^2+GIXOS.Qxy_px.^2);
geo.DSbetaHW = mean(GIXOS.tt(2:end) - GIXOS.tt(1:end-1))/2;

% now sum roiHW for the 2thetaHW
for idx = 1:length(tth_idx)
    GIXOS.GIXOS_raw(:,idx) = sum(importdata.Intensity(:,tth_idx(idx)-mat_roiHW:tth_idx(idx)+mat_roiHW),2);
    GIXOS.GIXOS_bkg(:,idx) = sum(importbkg.Intensity(:,tth_idx(idx)-mat_roiHW:tth_idx(idx)+mat_roiHW),2);
end

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
for idx = 1:length(tth_idx)
    GIXOS.GIXOS(:,idx)= (GIXOS.GIXOS_raw(:,idx) - GIXOS.GIXOS_bkg(:,idx) - GIXOS.bulk_bkg_apply(:,idx)).*GIXOS.fdtt;
    GIXOS.error(:,idx)= sqrt(sqrt(abs(GIXOS.GIXOS_raw(:,idx))).^2 + sqrt(abs(GIXOS.GIXOS_bkg(:,idx))).^2) .*GIXOS.fdtt;
    GIXOS.bkg(idx) = 0;
end
%%
% GIXOS
close(findobj('name','raw'));
fig_refl=figure('name','raw');
plot(GIXOS.Qz, GIXOS.GIXOS,'ko','MarkerSize',3,'DisplayName','corrected data by chamber transmission bkg');
hold on
plot(GIXOS.Qz, GIXOS.GIXOS_raw,'r-','LineWidth',1.5, 'DisplayName', 'raw data');
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

%% create pseudo reflectivity R = GIXOS/(DS/RRF)*RF/transmission
GIXOS.fresnel = GIXOS_fresnel(GIXOS.Qz(:,1),geo.Qc);
GIXOS.transmission = GIXOS_Tsqr(GIXOS.Qz(:,1),geo.Qc, geo.energy, geo.alpha_i, geo.Ddet, geo.footprint);
GIXOS.dQz = GIXOS_dQz(GIXOS.Qz(:,1),geo.energy, geo.alpha_i, geo.Ddet, geo.footprint);
% integration over beta and phi properly
for idx = 1:length(geo.qxy0)
    [GIXOS.DS_RRF(:,idx), GIXOS.DS_term(:,idx), GIXOS.RRF_term(:,idx)] = calc_DS_RRF_integ(GIXOS.tt,geo.qxy0(idx),geo.energy/1000, geo.alpha_i, geo.RqxyHW, geo.DSqxyHW_real(idx), geo.DSbetaHW, tension, temperature, amin);
    % pseudo reflecitivity Qz, R, dR, dQ
    GIXOS.refl(:,:,idx) = [GIXOS.Qz (GIXOS.GIXOS(:,idx)-GIXOS.bkg(idx))./GIXOS.DS_RRF(:,idx).*GIXOS.fresnel(:,2)./GIXOS.transmission(:,4) GIXOS.error(:,idx)./GIXOS.DS_RRF(:,idx).*GIXOS.fresnel(:,2)./GIXOS.transmission(:,4) GIXOS.dQz(:,5)];
end
%% structure factor of simple liquid surface
GIXOS.SF = exp(-(intrinsic_roughness*GIXOS.Qz(:,1)).^2);

%% plot
close(findobj('name','RRF'));
op = figure('name','RRF','Position',[100,100,750,600]);
pa = axes('Parent',op);
hold on
for idx = 1:length(geo.qxy0)
    errorbar(GIXOS.refl(:,1,idx), GIXOS.refl(:,2,idx)./GIXOS.fresnel(:,2)/RFscaling, GIXOS.refl(:,3,idx)./GIXOS.fresnel(:,2)/RFscaling, 'o' ,'MarkerSize', 5, 'LineWidth',1, 'CapSize', 4, 'Color', colorset(idx,:),'DisplayName',[num2str(geo.qxy0(idx),'%.2f'),' ',char(197),'^{-1}']);        
end
plot(GIXOS.Qz(:,1), exp(-(roughness*GIXOS.Qz(:,1)).^2).*GIXOS.SF, '-b', 'LineWidth', 2, 'DisplayName', ['simulated R/R_F: ', num2str(tension*1000,'%d'), 'mN/m, ', num2str(temperature, '%d'), 'K']);
hold off
xlim([0 1.05]);
ylim([1e-6 10]);
xlabel(['Q_z (' char(197) '^-^1)'],'FontSize',18);
ylabel('R/R_F','FontSize',18);
set(gca, 'YScale', 'Log');
legend('location','SouthWest','box','off','FontSize',18);
pa.FontSize = 18;
pa.LineWidth = 2;
pa.XTick = 0:0.2:1;
pa.TickDir = 'out';
saveas(op, strcat(path_out,sample,'_',num2str(scan,'%05d'),'_RRF_multiQxy.jpg'));

%%
close(findobj('name','RRF uncorrected'));
op1 = figure('name','RRF uncorrected','Position',[100,100,750,600]);
pa1 = axes('Parent',op1);
hold on
for idx = 1:length(geo.qxy0)
    plot(GIXOS.Qz, exp(-(incorrect_roughness(idx)*GIXOS.Qz).^2).*GIXOS.SF, '-', 'LineWidth', 2, 'Color', colorset(idx,:));
end
for idx = 1:length(geo.qxy0)
    plot(GIXOS.refl(:,1,idx), GIXOS.refl(:,2,idx)./GIXOS.fresnel(:,2).*GIXOS.DS_RRF(:,idx)/GIXOS.DS_RRF(1,idx)/RFscaling, 'o' ,'MarkerSize', 5, 'Color', colorset(idx,:)); % without DS/(R/RF) correction
end
hold off
xlim([0 1.05]);
ylim([1e-6 10]);
xlabel(['Q_z (' char(197) '^-^1)'],'FontSize',18);
ylabel('R*/R*_0','FontSize',18);
set(gca, 'YScale', 'Log');
for idx = 1:length(geo.qxy0)
    legendtext{idx} =  ['\sigma_R = ', num2str(incorrect_roughness(idx), '%.1f'),' ',char(197)];
end
legend(legendtext, 'location','SouthWest','box','off','FontSize',18);
pa1.FontSize = 18;
pa1.LineWidth = 2;
pa1.XTick = 0:0.2:1;
pa1.TickDir = 'out';
saveas(op1, strcat(path_out,sample,'_',num2str(scan,'%05d'),'_RRF_multiQxy_uncorrected.jpg'));
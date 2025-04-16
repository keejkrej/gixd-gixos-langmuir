%% Qxy dependency display for thin film %%
% Chen 06.06.2023
% bkg subtraction
% bkg is extrapolated linearly

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
geo.Qc =0.0218;
geo.energy=15000; 
geo.alpha_i=0.07;
geo.Ddet=560.7;
geo.pixel = 0.075;
geo.footprint= 30;
geo.wavelength = 12404/geo.energy;
geo.qxy0 = [0.01, 0.015, 0.02:0.01:0.1]; % a group of qxy0
geo.qxy_bkg = [0.35:0.01:0.45]; % bkg region from the GISAXS data itself
geo.RqxyHW = 0.002 ; %[A^-1] resolution for XRR
geo.DSresHW = 0.003; % HW of the DS resolution at specular
geo.DSqxyHW = 2*geo.DSresHW; % HW of the region of interest in qxy0 for integration, set to 2* DS resolution (res = 0.003A^-1)
geo.qz_selected = [0.1, 0.15];%, 0.4]'; % [0.1, 0.2, 0.35, 0.4, 0.45]' for POPC, [0.1, 0.3, 0.35, 0.4, 0.45, 0.6]' for DPPE
%%%%%%%%%%%%%%%%%%%%%%
%% file
%%%%%%%%%%%%%%%%%%%%%%
% file
path = '/gpfs/current/processed/GID/';
path_out = '/gpfs/current/processed/PseudoXRR/';
sample = 'dopc';
scan = 15;
I0_sample_chamber = 1; % 

bkgsample = 'dopc';
bkgscan = 27;

kb = 1.381E-23; % Boltzmann constant, J/K
tension = (73-0.1)/1000; % tension, [N/m]
temperature = 293; % [K] % POPC 20degC, DPPE 22degC
amin = 5; % minimal wavelength cutoff of the surface CW
eta = kb*temperature/(tension*(2*pi))*10^20.*(geo.qz_selected).^2;
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
for idx = 1:length(tth)
    tth_idx(idx) = find(importdata.tth>tth(idx),1,'first');
end
for idx = 1:length(tth_bkg)
    tth_bkg_idx(idx) = find(importdata.tth>tth_bkg(idx),1,'first');   % bkg tth in the GISAXS
end
tth_roiHW = rad2deg(geo.DSqxyHW .* geo.wavelength / 2 / pi ./ cosd(tth/2));   % region of interest of the tth
mat_roiHW = floor(tth_roiHW / geo.tth_step);
tth_roiHW_real = mat_roiHW*geo.tth_step;
geo.DSqxyHW_real = deg2rad(tth_roiHW_real)/2*4*pi/geo.wavelength .*cosd(tth/2);
%%
% arrange data structure
GIXOS.tt = importdata.tt;
geo.DSbetaHW = mean(GIXOS.tt(2:end) - GIXOS.tt(1:end-1))/2;
for idx = 1:length(tth)
    GIXOS.Qxy(:,idx) = 2*pi/geo.wavelength * sqrt((cosd(GIXOS.tt).*sind(tth(idx))).^2+(cosd(geo.alpha_i)-cosd(GIXOS.tt).*cosd(tth(idx))).^2);
end
GIXOS.Qz = 2*pi/geo.wavelength*(sind(GIXOS.tt)+sind(geo.alpha_i));
GIXOS.Q = sqrt(GIXOS.Qz.^2+GIXOS.Qxy.^2);
for idx = 1:length(tth)
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
%% bkg fit with Q
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

GIXOS.GIXOS= (GIXOS.GIXOS_raw - GIXOS.GIXOS_bkg - GIXOS.bulk_bkg_apply ).*GIXOS.fdtt;
GIXOS.error= sqrt(sqrt(abs(GIXOS.GIXOS_raw)).^2 + sqrt(abs(GIXOS.GIXOS_bkg)).^2) .*GIXOS.fdtt;
%% GIXOS
close(findobj('name','raw'));
fig_qxy=figure('name','raw');
hold on
for idx = 1:size(GIXOS.GIXOS, 2)
    plot(GIXOS.Qz, GIXOS.GIXOS(:, idx),'-','LineWidth',1.5,'DisplayName',['Q_{xy,0} =',num2str(geo.qxy0(idx), '%.2f'),'A^{-1}']);
end
yline(0,'k-.', 'LineWidth', 1.5, 'DisplayName', '0-line');
hold off
xlim([0 1.05]);
ax=gca;
xlabel(['Q_z [' char(197) '^-^1]'],'FontSize',12);
ylabel('GIXOS','FontSize',12);
ax.XTick=0:0.2:1;
ax.FontSize = 12;
ax.LineWidth = 2;
ax.TickDir = 'out';
legend('location','NorthEast','box','off');

close(findobj('name','raw Qxy'));
fig_reflqxy=figure('name','raw Qxy');
hold on
for idx = 1:length(geo.qz_selected)
    plotrowidx = find(GIXOS.Qz<=geo.qz_selected(idx),1,'last');
    plot(GIXOS.Qxy(plotrowidx,:), mean(GIXOS.GIXOS(plotrowidx-3:plotrowidx+3, :),1)*100^(idx-1),'o:','MarkerSize',4,'LineWidth',1.5,'DisplayName',['Q_z =',num2str(GIXOS.Qz(plotrowidx), '%.2f'),'A^{-1}']);
end
hold off;
xlim([0.015, 1]);
ax=gca;
xlabel(['Q_x_y [' char(197) '^-^1]'],'FontSize',12);
ylabel('GIXOS','FontSize',12);
set(gca, 'XScale','Log','YScale', 'Log');
ax.FontSize = 12;
ax.LineWidth = 2;
ax.TickDir = 'out';
legend('location','NorthEast','box','off','FontSize',12);

%% fit to get kapa
for idx = 1:length(geo.qz_selected)
    rowidx = find(GIXOS.Qz<=geo.qz_selected(idx),1,'last');
    GIXOS.GIXOS_kapafit(idx,:) = mean(GIXOS.GIXOS(rowidx-1:rowidx+1, :),1);
    params.beta_array(idx) = GIXOS.tt(rowidx);
end
params.energy = geo.energy/1000;
params.alpha = geo.alpha_i;
params.DSqxy_HWHM = geo.DSqxyHW_real;
params.DSbeta_HWHM = geo.DSbetaHW;
params.temp = temperature;
params.tension = tension;
params.amin = amin;

tcell={ 'kapa','kT'};

% customize the initial values (coeff), lower (lb) and the upper boundary (ub)
% for the parameters to be fit.
% sequence of the coeff, lb, ub follows the table of parameter "tcell"
coeff= [ 15 ];
lb   = [  0 ];
ub = [  50 ];

% customize linear constraints and non-linear constraints (null), A*x<=b
% for different sigma for air-tail, tail-CG, CG-head
A=[];
b=[];
Aeq=[];
beq=[];
nonlcon=[];

% fit options defined as:
options=optimoptions('fmincon','Display','iter','MaxIter',30000,'MaxFunEvals',500000,'TolFun',1e-8,'TolX',1e-8,'TolCon',1e-8,'Algorithm','interior-point');
% assign the fitting function as f using variables given in the "coeff" vector
f = @(coeff) fitfun_eCWM(geo.qxy0, GIXOS.GIXOS_kapafit, coeff, params);
% do fitting, result of variables into z, output hessian matrix
[z,fval,exitflag,output,lambda,grad,hessian]=fmincon(f,coeff,A,b,Aeq,beq,lb,ub,nonlcon,options);

% for matlab without financial toolbox, manually calculate it
cov=inv(hessian(:,:));
for m=1:size(cov,1)
    sigma(m)=sqrt(cov(m,m));
end;
err=sigma';
clear m;

kapa = z;  % [kT]
kapa_deviation = err;
assume_kapa = [kapa-kapa_deviation, kapa+kapa_deviation]; % assumed kapa
zeta = sqrt(kapa*kb*temperature/tension)*10^10; % [A]
%% diffuse scattering (DS)'s dependency on Qxy
clear model assume_model;
model.tt = ones(length(geo.qz_selected),1);
model.Qz = ones(length(geo.qz_selected),1);
for idx = 1:length(geo.qz_selected)
    rowidx = find(GIXOS.Qz<=geo.qz_selected(idx),1,'last');
    model.tt(idx) = GIXOS.tt(rowidx);
    model.Qz(idx) = GIXOS.Qz(rowidx);
    model.Qxy(idx,:) = GIXOS.Qxy(rowidx, :);
end

% models using assume kapa
assume_model(1) = model;
assume_model(2) = model;
% CWM surface
CWM_model = model;
times_start = posixtime(datetime('now'));
for idx = 1:size(geo.qxy0,2)
    [model.DS_RRF(:,idx), model.DS_term(:,idx), model.RRF_term(:,idx)] = calc_film_DS_RRF_integ_approx(model.tt,geo.qxy0(idx),geo.energy/1000, geo.alpha_i, geo.RqxyHW, geo.DSqxyHW_real(idx), geo.DSbetaHW, tension, temperature, kapa, amin);
    [assume_model(1).DS_RRF(:,idx), assume_model(1).DS_term(:,idx), assume_model(1).RRF_term(:,idx)] = calc_film_DS_RRF_integ_approx(model.tt,geo.qxy0(idx),geo.energy/1000, geo.alpha_i, geo.RqxyHW, geo.DSqxyHW_real(idx), geo.DSbetaHW, tension, temperature, assume_kapa(1), amin);
    [assume_model(2).DS_RRF(:,idx), assume_model(2).DS_term(:,idx), assume_model(2).RRF_term(:,idx)] = calc_film_DS_RRF_integ_approx(model.tt,geo.qxy0(idx),geo.energy/1000, geo.alpha_i, geo.RqxyHW, geo.DSqxyHW_real(idx), geo.DSbetaHW, tension, temperature, assume_kapa(2), amin);
    [CWM_model.DS_RRF(:,idx), CWM_model.DS_term(:,idx), CWM_model.RRF_term(:,idx)] = calc_CWM_DS_RRF_integ(model.tt,geo.qxy0(idx),geo.energy/1000, geo.alpha_i, geo.RqxyHW, geo.DSqxyHW_real(idx), geo.DSbetaHW, tension, temperature, amin);
end
fprintf('time spent: %.2f s\n', posixtime(datetime('now')) - times_start);

%%
close(findobj('name','model Qxy'));
fig_qxy=figure('name','model Qxy','Position',[50, 50, 900, 900]);
hold on
for idx = 1:length(model.Qz)
    plotrowidx = find(GIXOS.Qz<=geo.qz_selected(idx),1,'last');
    plot(GIXOS.Qxy(plotrowidx,:), GIXOS.GIXOS_kapafit(idx,:)*100^(idx-1),'o','MarkerSize',4,'MarkerEdgeColor',colorset(idx,:),'LineWidth',1.5);
    legendtext{idx} = ['Q_z =',num2str(GIXOS.Qz(plotrowidx), '%.2f'),char(197),'^{-1}'];
end
for idx = 1:length(model.Qz)
    plotrowidx = find(GIXOS.Qz<=geo.qz_selected(idx),1,'last');
    plot(model.Qxy(idx,:), (model.DS_term(idx,:)./model.DS_term(idx,2).*GIXOS.GIXOS_kapafit(idx,2))*100^(idx-1),'LineWidth',1.5,'Color',colorset(idx,:));
    plot(assume_model(1).Qxy(idx,:), (assume_model(1).DS_term(idx,:)./assume_model(1).DS_term(idx,2).*GIXOS.GIXOS_kapafit(idx,2))*100^(idx-1),':','LineWidth',1.5,'Color',colorset(idx,:));
    plot(assume_model(2).Qxy(idx,:), (assume_model(2).DS_term(idx,:)./assume_model(2).DS_term(idx,2).*GIXOS.GIXOS_kapafit(idx,2))*100^(idx-1),':','LineWidth',1.5,'Color',colorset(idx,:));
    plot(CWM_model.Qxy(idx,:), (CWM_model.DS_term(idx,:)./CWM_model.DS_term(idx,2).*GIXOS.GIXOS_kapafit(idx,2))*100^(idx-1),'k-.','LineWidth',0.5 )
end
hold off;
xlim([0.008, 0.2]);
ylim([10 1e14]);
ax=gca;
xlabel(['Q_x_y [' char(197) '^-^1]'],'FontSize',14);
ylabel('I_{DS}','FontSize',14);
set(gca, 'XScale','Log','YScale', 'Log');
ax.FontSize = 14;
ax.LineWidth = 2;
ax.TickDir = 'out';
legend(legendtext,'location','NorthWest','box','off','FontSize',14);
title(['k_c:', num2str(kapa,'%.0f'),'\pm',num2str(kapa_deviation,'%.0f'),'kbT, r_{min}: ', num2str(amin,'%.1f'),char(197)]);
%saveas(fig_refl, strcat(path_out,sample,'_',num2str(scan,'%05d'),'_Qxy.jpg'));

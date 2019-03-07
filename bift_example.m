clear all
close all

%%
fid = fopen('example_data.dat');
i_data = textscan(fid, '', 'commentstyle', '#');
q = i_data{1};
iq_exp = i_data{2};
iq_std = i_data{3};

% remove q = 0 point for stability:
q(1) = [];
iq_exp(1) = [];
iq_std(1) = [];

fid = fopen('pr_result_gnom.out');
gnom_raw = textscan(fid, '', 'commentstyle', '#');
r_gnom = gnom_raw{1};
pr_gnom = gnom_raw{2};
prerr_gnom = gnom_raw{3};

fid = fopen('pr_result_raw.ift');
gnom_raw = textscan(fid, '', 'commentstyle', '#');
r_raw = gnom_raw{1};
pr_raw = gnom_raw{2};
prerr_raw = gnom_raw{3};

[rgrid, P, Perr] = bift( q, iq_exp, iq_std );

%%
figure(1);
clf;

subplot(211)
hold on
errorbar(q, iq_exp, iq_std)

% set(gca, 'xscale', 'log')
% set(gca, 'yscale', 'log')

xlim([0.01, 1])
xlabel('q, 1/A')
ylabel('I(q)')
title('Mock data (PDB: 2A3G)')

subplot(212); hold on
plot(r_gnom, pr_gnom)
plot(r_raw, pr_raw)
errorbar(rgrid, P, Perr)
legend('GNOM', 'RAW', 'matlab bift')
xlabel('r, A')
ylabel('P(r)')
title('IFT results')
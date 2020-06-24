figure(1); clf;
col = [12, 123, 220] / 255;
linew = 2;
fsize = 20;
export_plots = true;

%Data fidelity vs L:
load('results/z_dim_paramSelect_fMNIST_class034.mat');
subplot(1,3,1);
h = errorbar(z_dim, -nll_final, -nll_final_std);
set(h,'linewidth',linew,'color',col); grid on;
axis([z_dim(1) z_dim(end) -35 -20]);
title('Step 1: select latent space dimension','fontsize',fsize);
xlabel('$$K+L$$','interpreter','latex');
ylabel('$$\mathcal{D}$$','interpreter','latex');

%Causal objective vs K:
load('results/alphaFinal_paramSelect_fMNIST_class034.mat');
subplot(1,3,2);
h = errorbar(alpha_dim, -ce_finalLam, -ce_finalLam_std);
set(h,'linewidth',linew,'color',col); grid on;
axis([alpha_dim(1)-0.1, alpha_dim(end)+0.1 0.98 1.05]);
title('Steps 2-3: increment causal factors','fontsize',fsize);
xlabel('$$K$$','interpreter','latex')
ylabel('$$\mathcal{C}$$','interpreter','latex')

%Data fidelity vs lambda for selected N_alpha
load('results/alpha2_paramSelect_fMNIST_class034.mat');
subplot(1,3,3);
h = errorbar(lambda, -nll_final, -nll_final_std);
set(h,'linewidth',linew,'color',col); grid on;
title('Steps 2-3: adjust \lambda','fontsize',fsize);
axis([min(lambda) max(lambda) -29 -20]);
xlabel('$$\lambda$$','interpreter','latex');
ylabel('$$\mathcal{D}$$','interpreter','latex');
set(gca,'XScale','log');

for i = 1:3
  subplot(1,3,i);
  a = gca;
  a.XAxis.FontSize = fsize;
  a.YAxis.FontSize = fsize;
  set(gca,'fontname','Times New Roman');
end
set(gcf,'color','white');
if export_plots
  fprintf('Exporting...');
  export_fig('./figs/fig_tuning_fmnist.pdf')
  saveas(gcf,'./figs/fig_tuning_fmnist.svg');
  fprintf('done!\n');
end
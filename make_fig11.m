%{
    make_fig11.py
    
    Reproduces Figure 11 in O'Shaughnessy et al., 'Generative causal
    explanations of black-box classifiers,' Proc. NeurIPS 2020: partial details
    of parameter tuning procedure for MNIST 3/8 example used to select K, L,
    and lambda.

    Note: this script uses pre-saved results in results/tuning_mnist38_*.mat.
    These .mat files contain additional information from the parameter turning
    process shown.
%}

figure(1); clf;
col = [12, 123, 220] / 255;
linew = 2;
fsize = 20;
export_plots = true;

%Data fidelity vs L:
load('results/tuning_mnist38_selectz.mat');
subplot(1,3,1);
h = errorbar(z_dim, -nll_final, -nll_final_std);
set(h,'linewidth',linew,'color',col); grid on;
axis([z_dim(1) z_dim(end) -42 -26]);
title('Step 1: select latent space dimension','fontsize',fsize);
xlabel('$$K+L$$','interpreter','latex');
ylabel('$$\mathcal{D}$$','interpreter','latex');

%Causal objective vs K:
load('results/tuning_mnist38_selectK.mat');
subplot(1,3,2);
h = errorbar(alpha_dim, -ce_finalLam, -ce_finalLam_std);
set(h,'linewidth',linew,'color',col); grid on;
axis([alpha_dim(1)-0.1, alpha_dim(end)+0.1 0.65 0.70]);
title('Steps 2-3: increment causal factors','fontsize',fsize);
xlabel('$$K$$','interpreter','latex')
ylabel('$$\mathcal{C}$$','interpreter','latex')

%Data fidelity vs lambda for selected N_alpha
load('results/tuning_mnist38_selectlambda.mat');
subplot(1,3,3);
h = errorbar(lambda, -nll_final, -nll_final_std);
set(h,'linewidth',linew,'color',col); grid on;
title('Steps 2-3: adjust \lambda','fontsize',fsize);
axis([min(lambda) max(lambda) -36 -26]);
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
  export_fig('./figs/fig_11.pdf')
  saveas(gcf,'./figs/fig_11.svg');
  fprintf('done!\n');
end
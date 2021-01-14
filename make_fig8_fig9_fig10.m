%{
    make_fig8_fig9_fig10.m
    
    Reproduces Figures 8-10 in O'Shaughnessy et al., 'Generative causal
    explanations of black-box classifiers,' Proc. NeurIPS 2020: empirical
    results for causal/information flow objectives with linear/gaussian
    generative map and toy classifier.

    Note: this plots the outputs of make_fig8.py and make_fig9_fig10.py.
%}

export_plots = true;

%% Figure 8 - linear classifier, objective variants
load('results/fig8.mat');
figure(1); clf;
fsize = 28;
clim_ce = [min(CEs(:)),max(CEs(:))];
% subplot 1 - causal effect (independent/unconditional)
subplot(141);
imagesc(thetas_alpha/pi*180, thetas_beta/pi*180, CEs(:,:,1).', clim_ce);
set(gca,'ydir','normal');
title('$$\mathcal{C}_{iu}$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% subplot 3 - causal effect (independent/conditional)
subplot(142);
imagesc(thetas_alpha/pi*180, thetas_beta/pi*180, CEs(:,:,2).', clim_ce);
set(gca,'ydir','normal');
title('$$\mathcal{C}_{ic}$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% subplot 4 - causal effect (joint/unconditional)
subplot(143);
imagesc(thetas_alpha/pi*180, thetas_beta/pi*180, CEs(:,:,3).', clim_ce);
set(gca,'ydir','normal');
title('$$\mathcal{C} = \mathcal{C}_{ju}$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% subplot 5 - causal effect (joint/conditional)
subplot(144);
imagesc(thetas_alpha/pi*180, thetas_beta/pi*180, CEs(:,:,4).', clim_ce);
set(gca,'ydir','normal');
title('$$\mathcal{C}_{jc}$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% formatting
for i = 1:4
  subplot(1,4,i);
  set(gca, 'fontsize', fsize, ...
    'xtick', 0:22.5:180, ...
    'xticklabel', {'0','','45','','90','','135','','180'}, ...
    'ytick', 0:22.5:180, ...
    'yticklabel', {'0','','45','','90','','135','','180'});
  xlabel('$$\theta(w_\alpha)$$','interpreter','latex','fontsize',fsize-4);
  ylabel('$$\theta(w_\beta)$$','interpreter','latex','fontsize',fsize-4);
  a = gca;
  a.XAxis.FontSize = fsize-8;
  a.YAxis.FontSize = fsize-8;
  set(gca,'fontname','Times New Roman');
  axis square; grid on; colorbar('fontsize',fsize-8);
end
set(gcf,'color','white');
if export_plots
  fprintf('Exporting Figure 8...');
  export_fig('./figs/fig8.pdf')
  saveas(gcf,'./figs/fig8.svg');
  fprintf('done!\n');
end


%% Figure 9 - and classifier, individual components of objective
load('results/fig9.mat');
figure(2); clf;
fsize = 28;
allnce = [data.ce_iu(:);
  data.ce_ic(:);
  data.ce_ju(:);
  data.ce_jc(:)];
allcecomp = [data.nentropy(:);
  data.expcondentropy(:);
  data.adj_iu_ic(:);
  data.adj_iu_ju(:);
  data.adj_ju_jc(:);
  data.adj_ic_jc(:)];
clim_ce = [min(allnce),max(allnce)];
% subplot 1 - log-likelihood
subplot(251);
imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, data.loglik.');
set(gca,'ydir','normal');
title('Log-likelihood','fontweight','normal');
set(get(gca,'title'),'fontsize',fsize)
% subplot 2 - causal effect (independent/unconditional)
subplot(252);
imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, data.ce_iu.', clim_ce);
set(gca,'ydir','normal');
title('$$\mathcal{C}_{iu}$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% subplot 3 - causal effect (independent/conditional)
subplot(253);
imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, data.ce_ic.', clim_ce);
set(gca,'ydir','normal');
title('$$\mathcal{C}_{ic}$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% subplot 4 - causal effect (joint/unconditional)
subplot(254);
imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, data.ce_ju.', clim_ce);
set(gca,'ydir','normal');
title('$$\mathcal{C} = \mathcal{C}_{ju}$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% subplot 5 - causal effect (joint/conditional)
subplot(255);
imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, data.ce_jc.', clim_ce);
set(gca,'ydir','normal');
title('$$\mathcal{C}_{jc}$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% subplot 6 - negative entropy
subplot(2,6,7);
imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, data.nentropy.');
set(gca,'ydir','normal');
title({'$$-H(p(Y))$$'},'interpreter','latex');
% subplot 7 - expected conditional entropy
subplot(2,6,8);
imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, data.expcondentropy.');
set(gca,'ydir','normal');
title('$$E_{\alpha}[H(Y|\alpha)]$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% subplot 8 - adjustment from indep/uncond to indep/cond
clim_adj = [min([data.adj_iu_ic(:); data.adj_iu_ju(:); data.adj_ju_jc(:); data.adj_ic_jc(:)]), ...
  max([data.adj_iu_ic(:); data.adj_iu_ju(:); data.adj_ju_jc(:); data.adj_ic_jc(:)])];
subplot(2,6,9);
imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, data.adj_iu_ic.', clim_adj);
set(gca,'ydir','normal');
title('$$\mathcal{C}_{iu} \rightarrow \mathcal{C}_{ic}$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% subplot 9 - adjustment from indep/uncond to joint/uncond
subplot(2,6,10);
imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, data.adj_iu_ju.', clim_adj);
set(gca,'ydir','normal');
title('$$\mathcal{C}_{iu} \rightarrow \mathcal{C}$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% subplot 10 - adjustment from joint/uncond to joint/cond
subplot(2,6,11);
imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, data.adj_ju_jc.', clim_adj);
set(gca,'ydir','normal');
title('$$\mathcal{C}_{ju} \rightarrow \mathcal{C}_{jc}$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% subplot 11 - adjustment from ind/cond to joint/cond
subplot(2,6,12);
imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, data.adj_ic_jc.', clim_adj);
set(gca,'ydir','normal');
title('$$\mathcal{C}_{ic} \rightarrow \mathcal{C}_{jc}$$','interpreter','latex');
set(get(gca,'title'),'fontsize',fsize)
% formatting
subplot(251);
for i = 1:5
  subplot(2,5,i);
  set(gca, 'fontsize', fsize, ...
    'xtick', 0:22.5:180, ...
    'xticklabel', {'0','','45','','90','','135','','180'}, ...
    'ytick', 0:22.5:180, ...
    'yticklabel', {'0','','45','','90','','135','','180'});
  xlabel('$$\theta(w_{\alpha_1})$$','interpreter','latex','fontsize',fsize-8);
  ylabel('$$\theta(w_{\alpha_2})$$','interpreter','latex','fontsize',fsize-8);
  a = gca;
  a.XAxis.FontSize = fsize-12;
  a.YAxis.FontSize = fsize-12;
  set(gca,'fontname','Times New Roman');
  axis square; grid on; colorbar('fontsize',fsize-12);
end
for i = 7:12
  subplot(2,6,i);
  set(gca, 'fontsize', fsize, ...
    'xtick', 0:22.5:180, ...
    'xticklabel', {'0','','45','','90','','135','','180'}, ...
    'ytick', 0:22.5:180, ...
    'yticklabel', {'0','','45','','90','','135','','180'});
  xlabel('$$\theta(w_{\alpha_1})$$','interpreter','latex','fontsize',fsize-8);
  ylabel('$$\theta(w_{\alpha_2})$$','interpreter','latex','fontsize',fsize-8);
  a = gca;
  a.XAxis.FontSize = fsize-12;
  a.YAxis.FontSize = fsize-12;
  set(gca,'fontname','Times New Roman');
  axis square; grid on; colorbar('fontsize',fsize-12);
end
set(gcf,'color','white');
if export_plots
  fprintf('Exporting Figure 9...');
  export_fig('./figs/fig9.pdf');
  saveas(gcf,'./figs/fig9.svg');
  fprintf('done!\n');
end


%% surface plots - two values of lambda
figure(3); clf;
fsize = 28;
lambdas = [0.0001 0.001];
for i = 1:2
  lambda = lambdas(i);
  obj_iu = data.ce_iu.' + lambda*data.loglik.';
  obj_ic = data.ce_ic.' + lambda*data.loglik.';
  obj_ju = data.ce_ju.' + lambda*data.loglik.';
  obj_jc = data.ce_jc.' + lambda*data.loglik.';
  all_objs = [obj_iu(:); obj_ic(:); obj_ju(:); obj_jc(:)];
  clim = [-1, 0.65];
  % subplot 1 - causal effect (independent/unconditional)
  subplot(2,4,4*(i-1)+1)
  imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, obj_iu, clim);
  title({'$$\mathcal{C}_{iu} + \lambda \cdot \mathcal{D}$$',sprintf('$$(\\lambda=10^{%d})$$', log10(lambda))}, ...
    'interpreter','latex','fontsize',fsize);
  % subplot 2 - causal effect (independent/conditional)
  subplot(2,4,4*(i-1)+2)
  imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, obj_ic, clim);
  title({'$$\mathcal{C}_{ic} + \lambda \cdot \mathcal{D}$$',sprintf('$$(\\lambda=10^{%d})$$', log10(lambda))}, ...
    'interpreter','latex','fontsize',fsize);
  % subplot 3 - causal effect (joint/unconditional)
  subplot(2,4,4*(i-1)+3)
  imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, obj_ju, clim);
  title({'$$\mathcal{C} + \lambda \cdot \mathcal{D}$$',sprintf('$$(\\lambda=10^{%d})$$', log10(lambda))}, ...
    'interpreter','latex','fontsize',fsize);
  % subplot 4 - causal effect (joint/conditional)
  subplot(2,4,4*(i-1)+4)
  imagesc(thetas_alpha1/pi*180, thetas_alpha2/pi*180, obj_jc, clim);
  title({'$$\mathcal{C}_{jc} + \lambda \cdot \mathcal{D}$$',sprintf('$$(\\lambda=10^{%d})$$', log10(lambda))}, ...
    'interpreter','latex','fontsize',fsize);
end
% format
for i = 1:8
  subplot(2,4,i);
  set(gca,'ydir','normal');
  set(gca, 'fontsize', fsize, ...
    'xtick', 0:22.5:180, ...
    'xticklabel', {'0','','45','','90','','135','','180'}, ...
    'ytick', 0:22.5:180, ...
    'yticklabel', {'0','','45','','90','','135','','180'});
  xlabel('$$\theta(w_{\alpha_1})$$','interpreter','latex','fontsize',fsize-4);
  ylabel('$$\theta(w_{\alpha_2})$$','interpreter','latex','fontsize',fsize-4);
  a = gca;
  a.XAxis.FontSize = fsize-8;
  a.YAxis.FontSize = fsize-8;
  set(gca,'fontname','Times New Roman');
  axis square; grid on; colorbar('fontsize',fsize-8);
end
set(gcf,'color','white');
if export_plots
  fprintf('Exporting Figure 10...');
  export_fig ./figs/fig10.pdf
  saveas(gcf,'./figs/fig10.svg');
  fprintf('done!\n');
end


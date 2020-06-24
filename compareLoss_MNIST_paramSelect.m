objective = 'JOINT_UNCOND';
z_dim = 1:12;
lambda = 0.001;
alpha = 0;
numAvg = 200;
% foldersUse = dir(['./results/mnist_' objective '_zdim3_alpha3_*']);
ce_final = zeros(length(lambda),length(alpha));
nll_final = zeros(length(lambda),length(alpha));

ce_final_std = zeros(length(lambda),length(alpha));
nll_final_std = zeros(length(lambda),length(alpha));

results_folder = './images/JOINT_UNCOND_paramSelect_class38/';

legendCell_alpha = cellstr(num2str(alpha','\\alpha=%-d'));
legendCell_z = cellstr(num2str(z_dim','z=%-d'));
legendCell_lambda = cellstr(num2str(lambda','\\lambda=%.4f'));

figure(1);
subplot(2,length(lambda),1)
figure(2)
subplot(2,length(alpha),1)
for z_idx = 1:length(z_dim)

        %         sprintf('%.1f',lambda(l_idx))
        if lambda == 1
            folderUse = ['./results/class38/mnist_' objective '_zdim' num2str(z_dim(z_idx)) '_alpha' num2str(alpha) '_No100_Ni25_lam' sprintf('%.1f',lambda(l_idx)) '_class38'];
        else
            folderUse = ['./results/class38/mnist_' objective '_zdim' num2str(z_dim(z_idx)) '_alpha' num2str(alpha) '_No100_Ni25_lam' num2str(lambda) '_class38'];
        end
        fileUse_results = dir([folderUse '/results_*.mat']);
        load([folderUse '/' fileUse_results.name])
        
        loss_ce = data.loss_ce;
        loss_nll = data.loss_nll;
        
        ce_final(z_idx) = mean(loss_ce(end-numAvg:end));
        nll_final(z_idx) = mean(loss_nll(end-numAvg:end)/params.lam_ML);
        
        ce_final_std(z_idx) = std(loss_ce(end-numAvg:end));
        nll_final_std(z_idx) = std(loss_nll(end-numAvg:end)/params.lam_ML);
        
        figure(1);
        subplot(1,2,1);hold all;
        plot(loss_ce);
        subplot(1,2,2);hold all;
        plot(loss_nll/params.lam_ML);

end


figure(1);
subplot(1,2,1);ylabel('causal effect');

subplot(1,2,2);ylabel('NLL');
legend(legendCell_z);
saveas(gcf,[results_folder 'optimizationTrials_changeZeta_noCE.png']);
saveas(gcf,[results_folder 'optimizationTrials_changeZeta_noCE.fig']);

figure;errorbar(z_dim,nll_final,nll_final_std);
xlabel('Latent Space Dimension');
ylabel('NLL');
saveas(gcf,[results_folder 'NLLVary_changeZeta_noCE.png']);
saveas(gcf,[results_folder 'NLLVary_changeZeta_noCE.fig']);

foldersUse = dir(['./results/class38/selectAlpha__mnist_JOINT_UNCOND_zdim8_alpha*']);
class_num = 2;
for k = 1:length(foldersUse)
    fileUse = ['./results/class38/' foldersUse(k).name '/sweepLatentFactors_test.mat'];
    
    %fileUse = [foldersUse(k).name '/results_*.mat'];
    if exist(fileUse)
        load(fileUse)
        fileUse_results = dir(['./results/class38/' foldersUse(k).name '/results_*.mat']);
        load(['./results/class38/' foldersUse(k).name '/' fileUse_results.name])
        
        
        figure('Visible','off','Position',[100 100 1200 300]);
        subplot(1,4,1);
        plot(data.loss_ce);
        ylabel('Causal Effect');
        xlabel('Training Steps');
        xlim([0 8000])
        subplot(1,4,2);
        plot(data.loss_nll);
        xlim([0 8000])
        ylabel('NLL');
        xlabel('Training Steps');
        title(params.objective)
        subplot(1,4,3);
        plot(data.loss_nll_mse);
        xlim([0 8000])
        ylabel('NLL MSE');
        xlabel('Training Steps');
        title(num2str(params.lam_ML))
        subplot(1,4,4);
        plot(data.loss_nll_kld);
        xlim([0 8000])
        ylabel('NLL KLD');
        xlabel('Training Steps');
        saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/lossPlots.png']);
        saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/lossPlots.fig']);
        
        figure('Visible','off');
        stem(-I_flow);
        ylim([0 1]);
        xlabel('Latent Dimension');
        ylabel('Information Flow');
        title(['Information Flow per Dim - N_{alpha}: ' params.alpha_dim])
        saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/infoFlow.png']);
        saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/infoFlow.fig']);
        
        numEx = size(imgOut_rand,1);
        z_dim = size(imgOut_rand,2);
        t_idx = size(imgOut_rand,3);
        imgSize = size(imgOut_rand,4);
        t_vals = 1:1:t_idx;
        
        for m = 1:z_dim
            imgPlot_rand = zeros(numEx*imgSize,length(t_vals)*(imgSize));
            imgPlot_real = zeros(numEx*imgSize,length(t_vals)*(imgSize));
            imgPlot_diff_rand = zeros(numEx*imgSize,3*(imgSize));
            imgPlot_diff_rand2 = zeros(numEx*imgSize,3*(imgSize));
            imgPlot_diff_real = zeros(numEx*imgSize,3*(imgSize));
            imgPlot_diff_real2 = zeros(numEx*imgSize,3*(imgSize));
            if class_num ==2
                probPlot_rand = zeros(numEx,t_idx);
                probPlot_real = zeros(numEx,t_idx);
            else
                probPlot_rand = zeros(numEx,t_idx,3);
                probPlot_real = zeros(numEx,t_idx,3);
            end
            for n = 1:numEx
                img0_rand = reshape(imgOut_rand(n,m,floor(t_idx/2)-5,:,:),28,28);
                img1_rand = reshape(imgOut_rand(n,m,floor(t_idx/2),:,:),28,28);
                img2_rand = reshape(imgOut_rand(n,m,floor(t_idx/2)+5,:,:),28,28);
                img0_real = reshape(imgOut_real(n,m,floor(t_idx/2)-5,:,:),28,28);
                img1_real = reshape(imgOut_real(n,m,floor(t_idx/2),:,:),28,28);
                img2_real = reshape(imgOut_real(n,m,floor(t_idx/2)+5,:,:),28,28);
                
                obj_diff_rand = imshowpair(img0_rand,img1_rand,'diff');
                imgPlot_diff_rand((n-1)*imgSize+1:n*imgSize,1:imgSize,:) = 255-img1_rand*255;
                imgPlot_diff_rand((n-1)*imgSize+1:n*imgSize,imgSize+1:2*imgSize,:) = 255-obj_diff_rand.CData;
                imgPlot_diff_rand((n-1)*imgSize+1:n*imgSize,imgSize*2+1:3*imgSize,:) = 255-img0_rand*255;
                
                obj_diff_rand = imshowpair(img2_rand,img1_rand,'diff');
                imgPlot_diff_rand2((n-1)*imgSize+1:n*imgSize,1:imgSize,:) = 255-img1_rand*255;
                imgPlot_diff_rand2((n-1)*imgSize+1:n*imgSize,imgSize+1:2*imgSize,:) = 255-obj_diff_rand.CData;
                imgPlot_diff_rand2((n-1)*imgSize+1:n*imgSize,imgSize*2+1:3*imgSize,:) = 255-img2_rand*255;
                
                obj_diff_real = imshowpair(img0_real,img1_real,'diff');
                imgPlot_diff_real((n-1)*imgSize+1:n*imgSize,1:imgSize,:) = 255-img1_real*255;
                imgPlot_diff_real((n-1)*imgSize+1:n*imgSize,imgSize+1:2*imgSize,:) = 255-obj_diff_real.CData;
                imgPlot_diff_real((n-1)*imgSize+1:n*imgSize,imgSize*2+1:3*imgSize,:) = 255-img0_real*255;
                
                obj_diff_real = imshowpair(img2_real,img1_real,'diff');
                imgPlot_diff_real2((n-1)*imgSize+1:n*imgSize,1:imgSize,:) = 255-img1_real*255;
                imgPlot_diff_real2((n-1)*imgSize+1:n*imgSize,imgSize+1:2*imgSize,:) = 255-obj_diff_real.CData;
                imgPlot_diff_real2((n-1)*imgSize+1:n*imgSize,imgSize*2+1:3*imgSize,:) = 255-img2_real*255;
                %                 obj_diff_rand = imshowpair(img2_rand,img1_rand);
                %                 imgPlot_diff_rand((n-1)*imgSize+1:n*imgSize,imgSize+1:2*imgSize,:) = obj_diff_rand.CData;
                %                 obj_diff_real = imshowpair(img0_real,img1_real);
                %                 imgPlot_diff_real((n-1)*imgSize+1:n*imgSize,1:imgSize,:) = obj_diff_real.CData;
                %                 obj_diff_real = imshowpair(img2_real,img1_real);
                %                 imgPlot_diff_real((n-1)*imgSize+1:n*imgSize,imgSize+1:2*imgSize,:) = obj_diff_real.CData;
                
                
                t_count = 1;
                for t_use = t_vals
                    imgTemp_rand = reshape(imgOut_rand(n,m,t_use,:,:),28,28);
                    imgTemp_real = reshape(imgOut_real(n,m,t_use,:,:),28,28);
                    imgPlot_rand((n-1)*imgSize+1:n*imgSize,(t_count-1)*imgSize+1:t_count*imgSize) = imgTemp_rand;
                    imgPlot_real((n-1)*imgSize+1:n*imgSize,(t_count-1)*imgSize+1:t_count*imgSize) = imgTemp_real;
                    t_count = t_count+1;
                end
                if class_num == 2
                    probPlot_rand(n,:) = reshape(probOut_rand(n,m,:,1),1,t_idx);
                    probPlot_real(n,:) = reshape(probOut_real(n,m,:,1),1,t_idx);
                else
                    probPlot_rand(n,:,:) = reshape(probOut_rand(n,m,:,:),1,t_idx,3);
                    probPlot_real(n,:,:) = reshape(probOut_real(n,m,:,:),1,t_idx,3);
                end
            end
            figure('Visible','off');imagesc(imgPlot_rand);colormap('gray');title(['Rand ' params.objective ' z_{dim}:' num2str(m) ' \lambda:' num2str(params.lam_ML)]);
            xlabel('Latent Sweep');
            ylabel('Example');
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/causalEffect_rand_latent' num2str(m) '.png']);
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/causalEffect_rand_latent' num2str(m) '.fig']);
            figure('Visible','off');imagesc(imgPlot_real);colormap('gray');title(['Real ' params.objective ' z_{dim}:' num2str(m) ' \lambda:' num2str(params.lam_ML)]);
            xlabel('Latent Sweep');
            ylabel('Example');
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/causalEffect_real_latent' num2str(m) '.png']);
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/causalEffect_real_latent' num2str(m) '.fig']);
            
            figure('Visible','off','Position',[100,100,240,800]);imagesc(uint8(imgPlot_diff_rand));colormap('gray');title(['Rand ' params.objective ' z_{dim}:' num2str(m) ' \lambda:' num2str(params.lam_ML)]);
            xlabel('Latent Sweep');
            ylabel('Example');
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/causalEffect_diff_rand1_latent' num2str(m) '.png']);
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/causalEffect_diff_rand1_latent' num2str(m) '.fig']);
            figure('Visible','off','Position',[100,100,240,800]);imagesc(uint8(imgPlot_diff_rand2));colormap('gray');title(['Rand ' params.objective ' z_{dim}:' num2str(m) ' \lambda:' num2str(params.lam_ML)]);
            xlabel('Latent Sweep');
            ylabel('Example');
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/causalEffect_diff_rand2_latent' num2str(m) '.png']);
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/causalEffect_diff_rand2_latent' num2str(m) '.fig']);
            
            
            figure('Visible','off','Position',[100,100,240,800]);imagesc(uint8(imgPlot_diff_real));colormap('gray');title(['Real ' params.objective ' z_{dim}:' num2str(m) ' \lambda:' num2str(params.lam_ML)]);
            xlabel('Latent Sweep');
            ylabel('Example');
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/causalEffect_diff_real1_latent' num2str(m) '.png']);
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/causalEffect_diff_real1_latent' num2str(m) '.fig']);
            figure('Visible','off','Position',[100,100,240,800]);imagesc(uint8(imgPlot_diff_real2));colormap('gray');title(['Real ' params.objective ' z_{dim}:' num2str(m) ' \lambda:' num2str(params.lam_ML)]);
            xlabel('Latent Sweep');
            ylabel('Example');
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/causalEffect_diff_real2_latent' num2str(m) '.png']);
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/causalEffect_diff_real2_latent' num2str(m) '.fig']);
            
            figure('Visible','off');imagesc(probPlot_rand);title(['Rand ' params.objective ' z_{dim}:' num2str(m) ' \lambda:' num2str(params.lam_ML)]);
            xlabel('Latent Sweep');
            ylabel('Example');
            caxis([0 1]);
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/classifierOutput_rand_latent' num2str(m) '.png']);
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/classifierOutput_rand_latent' num2str(m) '.fig']);
            figure('Visible','off');imagesc(probPlot_real);title(['Real ' params.objective ' z_{dim}:' num2str(m) ' \lambda:' num2str(params.lam_ML)]);
            xlabel('Latent Sweep');
            ylabel('Example');
            caxis([0 1]);
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/classifierOutput_real_latent' num2str(m) '.png']);
            saveas(gcf,[foldersUse(k).folder '/' foldersUse(k).name '/classifierOutput_real_latent' num2str(m) '.fig']);
            test= 1;
            
            
            
        end
        fprintf([foldersUse(k).name '\n'])
        close all;
        test = 1;

        
    end
    
    
end
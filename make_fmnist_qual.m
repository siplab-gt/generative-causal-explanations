results_dir = '/Users/matthewoshaughnessy/Dropbox/fmnist/';
results_folder = 'selectAlpha_fmnist_JOINT_UNCOND_zdim6_alpha1_No100_Ni25_lam0.05_class79';

export = true;

nsamples = 8;
latent_sweep_idx = [4,7,10,13,16,19,22]; % 13 - original point
latent_dim = [1 2 3];
titles = {'$$\alpha_1$$', '$$\beta_1$$', '$$\beta_2$$', '$$\beta_3$$'};
border_width = 2;
spacing_width = 1;
col_scheme = 'binary'; % 'binary' or 'spectrum'
images = 'separate'; % 'separate' or 'combined'

if strcmpi(col_scheme,'spectrum')
  cols = cbrewer('div','RdBu',100);
  cols = cols(15:85,:);
else
  col0 = [255 194 10] / 255;
  col1 = [12 123 220] / 255;
end

files = dir(fullfile(results_dir,results_folder));
load(fullfile(results_dir,results_folder,files(contains({files.name},'results')).name));
load(fullfile(results_dir,results_folder,'sweepLatentFactors.mat'));

%%
nlatentsweep = length(latent_sweep_idx);
nlatentdim = length(latent_dim);
s1 = 28 + 2*border_width;
s2 = 28 + 2*border_width + 2*spacing_width;

if strcmpi(images,'combined')
  figure(1); set(gcf,'Position',[1 1 1164 410]); clf;
  ha = tight_subplot(1,nlatentdim,[0.01 0.01],[0.01 0.01],[0.08 0.01]);
end

for k = 1:nlatentdim
  
  if strcmpi(images,'combined')
    axes(ha(k));
  else
    figure(k); set(gcf,'color','white'); clf;
  end
  
  latent_factor_image = zeros(nsamples*s2,nlatentsweep*s2,3);
  for i = 1:nsamples
    for j = 1:nlatentsweep
      img_idxs = {i, latent_dim(k), latent_sweep_idx(j)};
      class_prob = probOut_real(img_idxs{:},1);
      if strcmpi(col_scheme,'spectrum')
        col_idx = round(class_prob*(size(cols,1)-1))+1;
        bordered_image = repmat(shiftdim(cols(col_idx,:),-1),[s1 s1 1]);
      elseif class_prob < 0.5
        bordered_image = repmat(shiftdim(col0,-1),[s1 s1 1]);
      else
        bordered_image = repmat(shiftdim(col1,-1),[s1 s1 1]);
      end
      image = repmat(squeeze(1-imgOut_real(img_idxs{:},:,:)),[1 1 3]);
      bordered_image(border_width+1:end-border_width,border_width+1:end-border_width,:) = image;
      bordered_spaced_image = ones(s2,s2,3);
      bordered_spaced_image(spacing_width+1:end-spacing_width,spacing_width+1:end-spacing_width,:) = bordered_image;
      latent_factor_image((i-1)*s2+1:i*s2,(j-1)*s2+1:j*s2,:) = bordered_spaced_image;
    end
  end
  
  imagesc(latent_factor_image); axis square; axis off;
  if false
    title(titles{k},'interpreter','latex','fontsize',20);
    for i = 1:nsamples
      t = annotation('textbox',[0.01,0.882-0.0779*i,0.1,0],'String',sprintf('Sample %d:',i));
      set(t,'fontname','Times New Roman','fontsize',16,'linestyle','none');
    end
    for i = 1:nlatentdim
      a = annotation('arrow',(i-1)*0.2302+[0.17,0.085],0.15*[1,1]);
      set(a,'linewidth',1.0,'headstyle','vback3','headlength',5);
      a = annotation('arrow',(i-1)*0.2302+[0.21,0.295],0.15*[1,1]);
      set(a,'linewidth',1.0,'headstyle','vback3','headlength',5);
      a = annotation('ellipse',[(i-1)*0.2302+0.186, 0.142, 0.0055, 0.015]);
      set(a,'linewidth',0.5,'facecolor','k');
      t = annotation('textbox',[(i-1)*0.2302+0.095,0.145,0.1,0],'String', ...
        sprintf('\\textit{decreasing} %s',titles{i}));
      set(t,'fontname','Times New Roman','fontsize',12,'linestyle','none', ...
        'fontangle','italic','interpreter','latex');
      t = annotation('textbox',[(i-1)*0.2302+0.217,0.145,0.1,0],'String', ...
        sprintf('\\textit{increasing} %s',titles{i}));
      set(t,'fontname','Times New Roman','fontsize',12,'linestyle','none', ...
        'fontangle','italic','interpreter','latex');
    end
  end
  if strcmpi(images,'separate')
    fprintf('Exporting figure for latent dim %d...\n', latent_dim(k));
    if export
      saveas(gcf,sprintf('figs/fig_fmnist_qual_%d.svg',latent_dim(k)));
      export_fig('-transparent',sprintf('figs/fig_fminst_qual_%d.pdf',latent_dim(k)))
    end
  end
  
end

if strcmpi(images,'combined')
  set(gcf,'color','white');
  if export
    export_fig('figs/fig_fmnist_qual.pdf');
  end
end
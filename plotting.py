import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
#import gif

"""
surface plot

INPUTS:
    - ax : axis to draw figure on
    - x : numpy array corresponding to ROWS of Z (displayed on x-axis)
          x[0] corresponds to Z[0,:] and x[end] corresponds to Z[end,:]
    - y : numpy array corresponding to COLUMNS of Z (displayed on y-axis)
          y[0] corresponds to Z[:,0] and y[end] corresponds to Z[:,end]
    - Z : image to plot
    - clim : color limits for image; default: [min(Z), max(Z)]
"""
def plotsurface(ax, x, y, Z, clim=None):
    x = x.flatten()
    y = y.flatten()
    deltax = x[1]-x[0]
    deltay = y[1]-y[0]
    extent = (np.min(x)+deltax/2,
              np.max(x)-deltax/2,
              np.min(y)+deltay/2,
              np.max(y)-deltay/2)
    if clim == None:
        clim = [np.min(Z), np.max(Z)]
    im = ax.imshow(np.transpose(Z),
                   origin='lower',
                   extent=extent,
                   vmin=clim[0],
                   vmax=clim[1])
    return im


"""
debugPlot - create frame
"""
#@gif.frame
def debugPlot_frame(X, Xhat, W, What, k, steps, debug, params, classifier, decoded_points):
    
    plt.figure(figsize=(24, 6), dpi=100)
    
    nalpha = params["alpha_dim"]
    nbeta = params["z_dim"] - params["alpha_dim"]
    
    plot_X = X[0:params["plot_batchsize"],:]
    plot_Y = classifier(torch.from_numpy(plot_X).float())[0].detach().numpy()
    plot_x1min, plot_x2min = 1.5 * plot_X.min(axis=0)
    plot_x1max, plot_x2max = 1.5 * plot_X.max(axis=0)
    
    print('Training step %d/%d: updating plot...' % (k, steps))
    plt.clf()
    cols = cm.get_cmap('plasma')
    
    # training data, samples, and learned What
    if params["decoder_net"] is "linGauss":
        plt.subplot(1,4,1)
        plot_Xhat = Xhat.detach().numpy()
        plt.scatter(plot_X[:,0], plot_X[:,1], c=plot_Y[:,0], alpha=0.5)
        plt.scatter(plot_Xhat[:,0], plot_Xhat[:,1], linewidth=0.5, c='c', marker='x', alpha=0.3)
        plot_linex = np.array([plot_x1min, plot_x1max])
        if params["classifier_net"] is "oneHyperplane":
            plot_w = W[:,0]
            plt.plot(plot_linex, -plot_w[0]/plot_w[1]*plot_linex, c='k', ls=':')
        elif params["classifier_net"] is "twoHyperplane":
            plot_w1 = W[:,0]
            plot_w2 = W[:,1]
            plt.plot(plot_linex, -plot_w1[0]/plot_w1[1]*plot_linex, c='k', ls=':')
            plt.plot(plot_linex, -plot_w2[0]/plot_w2[1]*plot_linex, c='k', ls=':')
        for i in range(nalpha+nbeta):
            plot_what = What.detach().numpy()[:,i]
            if i < nalpha:
                plt.plot(plot_linex, -plot_what[0]/plot_what[1]*plot_linex,
                         label=r'$\widehat{w}_%d$ ($\alpha_%d$)'%(i+1,i+1))
            else:
                plt.plot(plot_linex, -plot_what[0]/plot_what[1]*plot_linex, ls='--',
                         label=r'$\widehat{w}_%d$ ($\beta_%d$)'%(i+1,i-nalpha+1))
        plt.xlim((plot_x1min, plot_x1max))
        plt.ylim((plot_x2min, plot_x2max))
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
    
    # alpha and beta
    nstd = 2.
    cols = cm.get_cmap('viridis')
    naihat_vals = decoded_points["samples"].shape[2]
    z_dim = decoded_points["samples"].shape[3]
    for i in range(nalpha+nbeta):
        plt.subplot(nalpha+nbeta,4,i*4+2)
        for j in range(naihat_vals):
            X = decoded_points["samples"][:,:,j,i]
            mu = np.mean(X, axis=1)
            Sigma = 1./X.shape[1] * np.matmul(X, X.T)
            w, v = np.linalg.eig(Sigma)
            theta = -np.arctan(v[0,0]/v[1,0])/np.pi*180.
            ellipse = matplotlib.patches.Ellipse((mu[0],mu[1]),nstd*w[1],nstd*w[0],angle=theta,alpha=0.5,color=cols(float(j)/naihat_vals))
            if i < nalpha:
                plt.title(r'$p(x|\alpha_%d=\widehat{\alpha}_%d)$'%(i+1,i+1))
            else:
                plt.title(r'$p(x|\beta_%d=\widehat{\beta}_%d)$'%(i-nalpha+1,i-nalpha+1))
            plt.gca().add_patch(ellipse)
        plt.plot(np.mean(decoded_points["samples"][:,:,:,i],axis=1)[0,:],
                 np.mean(decoded_points["samples"][:,:,:,i],axis=1)[1,:],
                 c='k')
        plt.axis('equal')
        plt.gca().set(xlim=(-10,10),ylim=(-10,10))
        plt.grid(True)
    
    # loss
    plt.subplot(params["z_dim"],4,3)
    plt.plot(np.arange(k)+1, debug["loss"][:k], linewidth=0.5, label="total loss")
    plt.grid(True)
    plt.legend()
    plt.xlim((1, steps))
    plt.title("Training step %d/%d" % (k, steps))
    plt.subplot(3,4,7)
    plt.plot(np.arange(k)+1, debug["loss_nll"][:k], linewidth=0.5, label="likelihood")
    plt.grid(True)
    plt.legend()
    plt.xlim((1, steps))
    plt.subplot(3,4,11)
    plt.plot(np.arange(k)+1, debug["loss_ce"][:k], linewidth=0.5, label="causal effect")
    #plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.xlim((1, steps))
    
    # alignment of What
    if params["decoder_net"] is "linGauss":
        if params["classifier_net"] is "oneHyperplane":
            plt.subplot(2,4,4)
            for i in range(1,nalpha+nbeta+1):
                plt.plot(np.arange(k)+1, debug["cossim_w1what%d"%(i)][:k],
                         linewidth=2.0, label="$<w_1,\widehat{w}_%d>$"%(i))
            plt.grid(True)
            plt.legend()
            plt.xlim((1, steps))
            plt.ylim((-1, 1))
            plt.subplot(2,4,8)
            for i in range(1,nalpha+nbeta+1):
                for j in range(i+1,nalpha+nbeta+1):
                    plt.plot(np.arange(k)+1, debug["cossim_what%dwhat%d"%(i,j)][:k],
                         linewidth=2.0, label="$<\widehat{w}_%d,\widehat{w}_%d>$"%(i,j))
            plt.grid(True)
            plt.legend()
            plt.xlim((1, steps))
            plt.ylim((-1, 1))
        elif params["classifier_net"] is "twoHyperplane":
            plt.subplot(3,4,4)
            for i in range(1,nalpha+nbeta+1):
                plt.plot(np.arange(k)+1, debug["cossim_w1what%d"%(i)][:k],
                         linewidth=2.0, label="$<w_1,\widehat{w}_%d>$"%(i))
            plt.grid(True)
            plt.legend()
            plt.xlim((1, steps))
            plt.ylim((-1, 1))
            plt.subplot(3,4,8)
            for i in range(1,nalpha+nbeta+1):
                plt.plot(np.arange(k)+1, debug["cossim_w2what%d"%(i)][:k],
                         linewidth=2.0, label="$<w_2,\widehat{w}_%d>$"%(i))
            plt.grid(True)
            plt.legend()
            plt.xlim((1, steps))
            plt.ylim((-1, 1))
            plt.subplot(3,4,12)
            for i in range(1,nalpha+nbeta+1):
                for j in range(i+1,nalpha+nbeta+1):
                    plt.plot(np.arange(k)+1, debug["cossim_what%dwhat%d"%(i,j)][:k],
                         linewidth=2.0, label="$<\widehat{w}_%d,\widehat{w}_%d>$"%(i,j))
            plt.grid(True)
            plt.legend()
            plt.xlim((1, steps))
            plt.ylim((-1, 1))
    
    # draw complete plot
    plt.tight_layout()
    plt.show()
    plt.draw()
    plt.pause(0.01)

#@gif.frame
def latentFactorPlot_frame(X, params, classifier, xposterior_samps, aihat_vals, i):
    plt.figure(figsize=(18, 6), dpi=100)
    plt.clf()
    
    plot_X = X[0:params["plot_batchsize"],:]
    plot_Y = classifier(torch.from_numpy(plot_X).float())[0].detach().numpy()
    plot_x1min, plot_x2min = 1.5 * plot_X.min(axis=0)
    plot_x1max, plot_x2max = 1.5 * plot_X.max(axis=0)
                
    # plot samples
    for l in range(xposterior_samps.shape[4]):
        for k in range(3):
            plt.subplot(3,5,k*5+l+1)
            plt.cla()
            plt.scatter(plot_X[:,0], plot_X[:,1], c=plot_Y[:,0], s=0.1, alpha=0.5)
            plt.scatter(np.squeeze(xposterior_samps[0,:,i,k,l]),
                        np.squeeze(xposterior_samps[1,:,i,k,l]),
                        linewidth=0.5, c='k', marker='.')
            plt.grid(True)
            plt.xlim((-3, 3))
            plt.ylim((-3, 3))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(r'$p(x|\alpha_%d=%.2f)$' % (k+1, aihat_vals[i]))
            plt.tight_layout()
        plt.show()
        plt.draw()
        plt.pause(0.01)

def latentFactorPlot(X, params, classifier, xposterior_samps, aihat_vals):
    print('Creating plot...')
    frames = []
    for i in range(xposterior_samps.shape[2]):
        frame = latentFactorPlot_frame(X, params, classifier, xposterior_samps, aihat_vals, i)
        frames.append(frame)
    gif.save(frames, "results_samp.gif", duration=100)
    print('Done! Saved to results_samp.gif.')
    
def outline_mask(ax, mask, bounds=(0,1,0,1), color=(0,0,0,0.25)):
    # https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph
    
    x0, x1, y0, y1 = bounds
    
    # a vertical line segment is needed, when the pixels next to each other horizontally
    #   belong to diffferent groups (one is part of the mask, the other isn't)
    # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates 
    ver_seg = np.where(mask[:,1:] != mask[:,:-1])
    
    # the same is repeated for horizontal segments
    hor_seg = np.where(mask[1:,:] != mask[:-1,:])
    
    # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
    #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
    # in order to draw a discountinuous line, we add Nones in between segments
    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0]+1))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan,np.nan))
    
    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1]+1, p[0]))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan, np.nan))
    
    # now we transform the list into a numpy array of Nx2 shape
    segments = np.array(l)
    
    # now we need to know something about the image which is shown
    #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
    #   drawn with origin='lower'
    # with this information we can rescale our points
    segments[:,0] = x0 + (x1-x0) * segments[:,0] / mask.shape[1]
    segments[:,1] = y0 + (y1-y0) * segments[:,1] / mask.shape[0]
    
    # and now there isn't anything else to do than plot it
    ax.plot(segments[:,0], segments[:,1], color=color, linewidth=1)
    
    
    
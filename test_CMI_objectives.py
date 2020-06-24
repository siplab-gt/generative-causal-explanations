from CVAE import *

obj_sweep = ["IND_UNCOND", "IND_COND", "JOINT_UNCOND", "JOINT_COND"]
lambda_sweep = [0.000001]#, 0.00001, 0.0001]
randseed_sweep = [1, 2, 3, 4, 5, 6]
sweep_params = {"obj" : obj_sweep, "lambda" : lambda_sweep, "randseed" : randseed_sweep}

steps = 6000

results_shape = (steps, len(obj_sweep), len(lambda_sweep), len(randseed_sweep))
results = {}
results["yhat_min"]          = np.zeros(results_shape)
results["loss"]              = np.zeros(results_shape)
results["loss_ce"]           = np.zeros(results_shape)
results["loss_nll"]          = np.zeros(results_shape)
results["loss_nll_logdet"]   = np.zeros(results_shape)
results["loss_nll_quadform"] = np.zeros(results_shape)
results["cossim_w1what1"]    = np.zeros(results_shape)
results["cossim_w1what2"]    = np.zeros(results_shape)
results["cossim_w2what1"]    = np.zeros(results_shape)
results["cossim_w2what2"]    = np.zeros(results_shape)
results["cossim_what1what2"] = np.zeros(results_shape)

for i_obj, obj in enumerate(obj_sweep):
    for i_lam, lam in enumerate(lambda_sweep):    
        for i_randseed, randseed in enumerate(randseed_sweep):
            print("Objective %d/%d, lambda %d/%d, randseed %d/%d..." % \
                  (i_obj+1, len(obj_sweep),
                  i_lam+1, len(lambda_sweep),
                  i_randseed+1, len(randseed_sweep)))            
            trial_results = CVAE(steps = steps,
                                 lam_ML = lam, 
                                 objective = obj,
                                 decoder_net = 'nonLinGauss',
				 classifier_net = 'hyperplane',
				 randseed = randseed,
                                 save_output = False,
                                 debug_level = 0)
            for key in results.keys():
                results[key][:,i_obj,i_lam,i_randseed] = trial_results[key]
            print("Trial complete.")
            
print("All trials complete! Saving...")
timestamp = ''.join(re.findall(r'\d+', str(datetime.datetime.now())[:19]))
matfilename = 'results_' + timestamp + '.mat'
sio.savemat(matfilename, {'params' : sweep_params, 'data' : results})
print('Complete! Saved to ' + matfilename + '.')

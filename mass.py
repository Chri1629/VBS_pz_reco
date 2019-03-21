'''
Programm to evaluate the mass of the boson 

			Command line in order to compile from shell:
 			python mass.py -t 'ewk_signal_withreco_truth.npy' -i 'ewk_input.npy' -p "/home/christian/Scrivania/tesi/Risultati per tesi/rete_scelta/mse_three_hidden_layer_400"

Christian Uccheddu
'''



#	Import of useful libraries


import pylab as pl
import numpy as np  
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt 
import matplotlib as mp
from array import array



#>>>>>>>>>>>> FUNCTIONS USED IN THE PROGRAMME <<<<<<<<<<<<<<<<<<

#	Initialization of the parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help="Inserire il file di input")
parser.add_argument('-t', '--truth', type=str, required=True, help="Inserire il file dei valori veri")
parser.add_argument('-p', '--predictions', type=str, required=True, help="Inserire la directory in cui trovare le predizioni")
parser.add_argument('-ev', '--evaluate', action="store_true")

args = parser.parse_args()


def histo_plotter_predicted(predicted):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(predicted, bins=300, range = [70,90])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ M_{W}^{predicted} $', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.predictions + '/histogram_plot_predicted_mass.pdf', bbox_inches='tight')

def histo_plotter_estimator_mass_predicted(estimator):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(estimator, bins=300, range = [-0.5,0.5])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ \frac{M_{W}^{predicted}-M_{W}^{true}}{M_{W}^{true}} $', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.predictions + '/histogram_plot_predicted_mass_estimator.pdf', bbox_inches='tight')

def histo_plotter_estimator_mass_analytical(estimator):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(estimator, bins=300, range = [-0.5,0.5])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ \frac{M_{W}^{analytical}-M_{W}^{true}}{M_{W}^{true}} $', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.predictions + '/histogram_plot_analytical_mass_estimator.pdf', bbox_inches='tight')


def histo_plotter_analytic(analytic):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(analytic, bins=300, range = [70,90])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ M_{W}^{analytic} $', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.predictions + '/histogram_plot_analytic_mass.pdf', bbox_inches='tight')

def histo_plotter_true(true):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(true, bins=300, range = [70,90])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ M_{W}^{true} $', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.predictions + '/histogram_plot_true_mass.pdf', bbox_inches='tight')


def plot_2d_histo_1(predicted, true):  
    

    fig = plt.figure()
    ax = fig.add_subplot(111)

    H = ax.hist2d(predicted, true, bins=200, range = [[60,100],[60,100]] , cmap = "Blues")

    fig.colorbar(H[3], ax=ax, shrink=0.8, pad=0.01, orientation="vertical")
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$M_{W}^{predicted}$', size = 35) 
    plt.ylabel(r'$M_{W}^{true}$', size = 35) 
    plt.show()
    fig.savefig(args.predictions+ '/predicted_vs_true_mass_histogram2d_plot.pdf', bbox_inches='tight')


def plot_2d_histo_2(analytical, true):  
    

    fig = plt.figure()
    ax = fig.add_subplot(111)

    H = ax.hist2d(analytical, true, bins=200, range = [[60,100],[60,100]] , cmap = "Blues")

    fig.colorbar(H[3], ax=ax, shrink=0.8, pad=0.01, orientation="vertical")
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$M_{W}^{analytical}$', size = 35) 
    plt.ylabel(r'$M_{W}^{true}$', size = 35) 
    plt.show()
    fig.savefig(args.predictions+ '/analytical_vs_true_mass_histogram2d_plot.pdf', bbox_inches='tight')
    




#>>>>>>>>>>>> CORE OF THE PROGRAMME <<<<<<<<<<<<<<<<<<

#	Opening of the dataset, I'll take only the test dataset to compare it with neural network predictions
#   Importing first the values of neutrino

ewk_truth = np.load(args.truth)

W_mass_true = ewk_truth[200000:300000,0]
p_nu_x = ewk_truth[200000:300000,2]
p_nu_y = ewk_truth[200000:300000,3]
p_nu_z_analytical = ewk_truth[200000:300000,8]
Energy_nu_analytical = np.sqrt(p_nu_x**2+p_nu_y**2+p_nu_z_analytical**2)

p_nu_z_predicted = np.loadtxt(args.predictions+ '/predictions.txt', delimiter=',')
Energy_nu_predicted = np.sqrt(p_nu_x**2+p_nu_y**2+p_nu_z_predicted**2)

#   Importing the data for lepton

ewk_input = np.load(args.input)
Energy_l = ewk_input[200000:300000,1]
p_l_x = ewk_input[200000:300000,2]
p_l_y = ewk_input[200000:300000,3]
p_l_z = ewk_input[200000:300000,4]


#   Computation of W_mass in analytical way and with prediction

W_mass_predicted = np.sqrt((Energy_l+Energy_nu_predicted)**2-(p_nu_x+p_l_x)**2-(p_nu_y+p_l_y)**2-(p_nu_z_predicted+p_l_z)**2)
W_mass_analytical = np.sqrt((Energy_l+Energy_nu_analytical)**2-(p_nu_x+p_l_x)**2-(p_nu_y+p_l_y)**2-(p_nu_z_analytical+p_l_z)**2)
estimator1 = (W_mass_predicted-W_mass_true)/(W_mass_true)
estimator2 = (W_mass_analytical-W_mass_true)/(W_mass_true)


#   Some useful plots
histo_plotter_predicted(W_mass_predicted)
histo_plotter_analytic(W_mass_analytical)
histo_plotter_true(W_mass_true)
histo_plotter_estimator_mass_predicted(estimator1)
histo_plotter_estimator_mass_analytical(estimator2)

plot_2d_histo_1(W_mass_predicted, W_mass_true)
plot_2d_histo_2(W_mass_analytical, W_mass_true)


#   Computation of the efficiency of the network
efficiency_W_mass_analytic_reco = (abs(W_mass_predicted-W_mass_true))/(W_mass_true)
efficiency_W_mass_predicted_reco = (abs(W_mass_analytical-W_mass_true))/(W_mass_true)

efficiency_W_mass_analytic_reco = estimator1.mean()
efficiency_W_mass_predicted_reco = estimator2.mean()

#   Evaluating reconstructed masses:

W_mass_analytic_mean = W_mass_analytical.mean()
W_mass_analytic_std = W_mass_analytical.std()
rel_error_analytic = W_mass_analytic_std/W_mass_analytic_mean

W_mass_predicted_mean = W_mass_predicted.mean()
W_mass_predicted_std = W_mass_predicted.std()
rel_error_predicted = W_mass_predicted_std/W_mass_predicted_mean

#   Saving of useful values

if not args.evaluate:
    print(">>>>>>>>> SAVING HYPERPARAMETERS >>>>>>>>")
    f = open(args.predictions + "/W_mass_reco.txt", "w")
    f.write("efficiency_W_mass_analytic_reco: {0}\n".format(efficiency_W_mass_analytic_reco))
    f.write("efficiency_W_mass_predicted_reco: {0}\n".format(efficiency_W_mass_predicted_reco))
    f.write("W_mass_analytic_mean: {0}\n".format(W_mass_analytic_mean))
    f.write("W_mass_analytic_std: {0}\n".format(W_mass_analytic_std))    
    f.write("W_mass_predicted_mean: {0}\n".format(W_mass_predicted_mean))
    f.write("W_mass_predicted_std: {0}\n".format(W_mass_predicted_std))  
    f.write("rel_error_analytic: {0}\n".format(rel_error_analytic))
    f.write("rel_error_predicted: {0}\n".format(rel_error_predicted))     
    f.close()

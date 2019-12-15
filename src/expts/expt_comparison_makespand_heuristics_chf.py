# Make plots for this for the demo. Use this for about 100 sites
# so that the algorithms are scalable and you can obtain enough
# data for the purpose of plots. 
                                                                                                 
import sys, os                                                                                   
import numpy as np                                                                               
import matplotlib.pyplot as plt                                                                  
from matplotlib import rc                                                                        
from colorama import Fore, Back, Style                                                           
import argparse                                                                                  
sys.path.append('../lib')                                                                        
import problem_classic_horsefly as chf                                                           
import utils_graphics                                                                            
import utils_algo                                                                                
import scipy                                                                                     
import matplotlib as mpl                                                                         
import data_sets as ds                      
        
num_reps          = 10
num_pts_per_cloud = 10

# uni, clusunif, annulus, normal, spokes, grid
cloud_type        = 'uni'

mspans_greedy         = [] ; greedy_best_count    = 0
mspans_greedy_l1      = [] ; greedyl1_best_count  = 0
mspans_gincex         = [] ; gincex_best_count    = 0
mspans_gincoll        = [] ; gincoll_best_count   = 0
mspans_gincl1         = [] ; gincl1_best_count    = 0
mspans_k2means        = [] ; k2means_best_count   = 0
mspans_k2meansl1      = [] ; k2meansl1_best_count = 0
mspans_k3means        = [] ; k3means_best_count   = 0
mspans_k3meansl1      = [] ; k3meansk1_best_count = 0
mspans_tsp            = [] ; tsp_best_count       = 0 
mspans_tspl1          = [] ; tspl1_best_count     = 0
mspans_split_partition = [ ]; split_partition_best_count = 0

inithorseposn = np.asarray([0.5, 0.5])
phi           = 3.0

for i in range(num_reps):
    

    print "Rep: ", i
    #sites = list(np.random.rand(num_pts_per_cloud,2))
    sites = ds.genpointset(num_pts_per_cloud,cloud_type)
    assert len(sites) == num_pts_per_cloud, "Number of sites should be equal to num_pts_per_cloud "

    current_iter_best_mspan = np.inf
    current_iter_best_algo  = None

    #utils_algo.print_list(sites)
    #ds.plotpoints(sites)
    #sys.exit()

    algo_greedy_data = chf.algo_greedy(sites, inithorseposn, phi, 
                                       write_algo_states_to_disk_p = False   ,
                                       animate_schedule_p          = False   , 
                                       post_optimizer              = chf.algo_exact_given_specific_ordering)

    mspans_greedy.append(algo_greedy_data['tour_length_with_waiting_time_included'])
    print Fore.CYAN, algo_greedy_data, Style.RESET_ALL

    if mspans_greedy[-1] < current_iter_best_mspan:
        current_iter_best_mspan = mspans_greedy[-1]
        current_iter_best_algo  = 'greedy'

    #--------------------------------------------------------------------------------------------------------

    # algo_greedy_l1_data = chf.algo_greedy(sites, inithorseposn, phi, 
    #                                       write_algo_states_to_disk_p = False   ,
    #                                       animate_schedule_p          = False   , 
    #                                       post_optimizer              = chf.algo_approximate_L1_given_specific_ordering)

    # mspans_greedy_l1.append(algo_greedy_l1_data['tour_length_with_waiting_time_included'])
    # print Fore.CYAN, algo_greedy_l1_data, Style.RESET_ALL

    # if mspans_greedy_l1[-1] < current_iter_best_mspan:
    #     current_iter_best_mspan = mspans_greedy_l1[-1]
    #     current_iter_best_algo  = 'greedyl1'



    #--------------------------------------------------------------------------------------------------------
    algo_gincex_data = chf.algo_greedy_incremental_insertion(sites, inithorseposn, phi, 
                                                             write_algo_states_to_disk_p = False   ,
                                                             animate_schedule_p          = False   , 
                                                             post_optimizer              = chf.algo_exact_given_specific_ordering)

    mspans_gincex.append(algo_gincex_data['tour_length_with_waiting_time_included'])
    print Fore.CYAN, algo_gincex_data, Style.RESET_ALL

    
    if mspans_gincex[-1] < current_iter_best_mspan:
        current_iter_best_mspan = mspans_gincex[-1]
        current_iter_best_algo  = 'gincex'

    #--------------------------------------------------------------------------------------------------------
    #algo_gincoll_data = chf.algo_greedy_incremental_insertion(sites, inithorseposn, phi, 
    #                                                          write_algo_states_to_disk_p = False   ,
    #                                                          animate_schedule_p          = False   , 
    #                                                          post_optimizer              = None)

    #mspans_gincoll.append(algo_gincoll_data['tour_length_with_waiting_time_included'])
    #print Fore.CYAN, algo_gincoll_data , Style.RESET_ALL

    # if mspans_gincoll[-1] < current_iter_best_mspan:
    #     current_iter_best_mspan = mspans_gincoll[-1]
    #     current_iter_best_algo  = 'gincoll'


    #--------------------------------------------------------------------------------------------------------

    # algo_gincl1_data = chf.algo_greedy_incremental_insertion(sites, inithorseposn, phi, 
    #                                                          write_algo_states_to_disk_p = False   ,
    #                                                          animate_schedule_p          = False   , 
    #                                                          post_optimizer              = chf.algo_approximate_L1_given_specific_ordering)
    # mspans_gincl1.append(algo_gincl1_data['tour_length_with_waiting_time_included'])
    # print Fore.CYAN, algo_gincl1_data, Style.RESET_ALL

    # if mspans_gincl1[-1] < current_iter_best_mspan:
    #     current_iter_best_mspan = mspans_gincl1[-1]
    #     current_iter_best_algo  = 'gincl1'

    #--------------------------------------------------------------------------------------------------------
    algo_k2means_data = chf.algo_kmeans(sites, inithorseposn, phi, k=2,
                                        post_optimizer              = chf.algo_exact_given_specific_ordering)

    mspans_k2means.append(algo_k2means_data['tour_length_with_waiting_time_included'])
    print Fore.CYAN, algo_k2means_data, Style.RESET_ALL

    if mspans_k2means[-1] < current_iter_best_mspan:
        current_iter_best_mspan = mspans_k2means[-1]
        current_iter_best_algo  = 'k2means'



    #--------------------------------------------------------------------------------------------------------
    # algo_k2meansl1_data = chf.algo_kmeans(sites, inithorseposn, phi, k=2,
    #                                       post_optimizer  = chf.algo_approximate_L1_given_specific_ordering)

    # mspans_k2meansl1.append(algo_k2meansl1_data['tour_length_with_waiting_time_included'])
    # print Fore.CYAN, algo_k2meansl1_data, Style.RESET_ALL


    # if mspans_k2meansl1[-1] < current_iter_best_mspan:
    #     current_iter_best_mspan = mspans_k2meansl1[-1]
    #     current_iter_best_algo  = 'k2meansl1'

    #--------------------------------------------------------------------------------------------------------
    #algo_k3means_data = chf.algo_kmeans(sites, inithorseposn, phi, k=3,
    #                                    post_optimizer              = chf.algo_exact_given_specific_ordering)

    #mspans_k3means.append(algo_k3means_data['tour_length_with_waiting_time_included'])
    #print Fore.CYAN, algo_k3means_data, Style.RESET_ALL

    # if mspans_k3means[-1] < current_iter_best_mspan:
    #     current_iter_best_mspan = mspans_k3means[-1]
    #     current_iter_best_algo  = 'k3means'

    #--------------------------------------------------------------------------------------------------------
    # algo_k3meansl1_data = chf.algo_kmeans(sites, inithorseposn, phi, k=3,
    #                                       post_optimizer  = chf.algo_approximate_L1_given_specific_ordering)

    # mspans_k3meansl1.append(algo_k3meansl1_data['tour_length_with_waiting_time_included'])
    # print Fore.CYAN, algo_k3meansl1_data, Style.RESET_ALL

    # if mspans_k3meansl1[-1] < current_iter_best_mspan:
    #     current_iter_best_mspan = mspans_k3meansl1[-1]
    #     current_iter_best_algo  = 'k3meansl1'


    #--------------------------------------------------------------------------------------------------------
    algo_tsp_data = chf.algo_tsp_ordering_new(sites, inithorseposn, phi,
                                          post_optimizer  = chf.algo_exact_given_specific_ordering)
    mspans_tsp.append(algo_tsp_data['tour_length_with_waiting_time_included'])

    print Fore.CYAN, algo_tsp_data, Style.RESET_ALL
    
    if mspans_tsp[-1] < current_iter_best_mspan:
        current_iter_best_mspan = mspans_tsp[-1]
        current_iter_best_algo  = 'tsp'

    #--------------------------------------------------------------------------------------------------------
    # algo_tspl1_data = chf.algo_tsp_ordering(sites, inithorseposn, phi,
    #                                         post_optimizer  =  chf.algo_approximate_L1_given_specific_ordering)

    # mspans_tspl1.append(algo_tspl1_data['tour_length_with_waiting_time_included'])
    # print Fore.CYAN, algo_tspl1_data, Style.RESET_ALL
    #--------------------------------------------------------------------------------------------------------

    # algo_split_partition_data = chf.algo_split_partition(sites, inithorseposn, phi,
    #                                       post_optimizer  = chf.algo_exact_given_specific_ordering)
    # mspans_split_partition.append(algo_split_partition_data['tour_length_with_waiting_time_included'])

    # print Fore.CYAN, algo_split_partition_data, Style.RESET_ALL
    
    # if mspans_split_partition[-1] < current_iter_best_mspan:
    #     current_iter_best_mspan = mspans_split_partition[-1]
    #     current_iter_best_algo  = 'split_partition'


    #.....................................................
    if current_iter_best_algo == 'greedy':
        greedy_best_count += 1

    elif current_iter_best_algo == 'gincex':
        gincex_best_count += 1 
        
    elif current_iter_best_algo == 'k2means':
        k2means_best_count += 1
        
    elif current_iter_best_algo == 'tsp':
        tsp_best_count += 1
    
    #elif current_iter_best_algo == 'split_partition':
    #    split_partition_best_count += 1
    #.....................................................


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig0, ax0 = plt.subplots()
ax0.set_xlabel("Run Number", fontsize=15)
ax0.set_ylabel("Makespans of heuristics", fontsize=15)
ax0.set_xticks(range(0,num_reps,5))
ax0.grid(True, linestyle='--')
ax0.tick_params(labelsize=15)

#ax0.set_ylim([0.2,1.75])

lw = 2.0
ax0.plot(range(num_reps) , mspans_greedy    , 'o-', label=r"greedy" ,color="blue", lw=lw)
ax0.plot(range(num_reps) , mspans_gincex    , 'o-', label=r"gincex" ,color="orange", lw=lw)
ax0.plot(range(num_reps) , mspans_k2means   , 'o-', label=r"k2means",color="purple", lw=lw)
#ax0.plot(range(num_reps) , mspans_k3means   , 'o-', label=r"k3means",color="red", lw=lw)
ax0.plot(range(num_reps) , mspans_tsp       , 'o--', label=r"tsp"    ,color="green", lw=lw)
#ax0.plot(range(num_reps) , mspans_split_partition       , 'o-', label=r"splitpart"    ,color="black", lw=lw)

#ax0.plot(range(num_reps) , mspans_gincoll   , 'o-', label=r"gincoll",color="olive", lw=lw)
#ax0.plot(range(num_reps) , mspans_gincl1    , 'mo-', label=r"gincl1" ,color="brown", lw=lw)
#ax0.plot(range(num_reps) , mspans_k2meansl1 , 'o-', label=r"k2meansl1",color="limegreen", lw=lw)
#ax0.plot(range(num_reps) , mspans_k3meansl1 , 'o-', label=r"k3meansl1", color="tomato", lw=lw)
#ax0.plot(range(num_reps) , mspans_greedy_l1 , 'go-', label=r"greedyl1", lw=lw)
#ax0.plot(range(num_reps) , mspans_tspl1     , 'o-', label=r"tspl1"    , color="darkorange", lw=lw)

ax0.set_title("Number of Sites: " + str(num_pts_per_cloud)+ "      " +\
             r"$\varphi$: "     + str(phi)                         +\
             "\nCloud Type: "   + cloud_type, fontsize=21)
ax0.legend(bbox_to_anchor=(-0.12,0.8)  , loc='lower right', ncol=1, fancybox=True, shadow=True, prop={'size': 14})
plt.savefig('expt_comp_graph_mspan_nsites_' + str(num_pts_per_cloud)+\
            '_phi_'                   + str(phi) +\
            '_cloudtype_'             + cloud_type + '.png',\
            bbox_inches = 'tight', dpi=200)
print Fore.GREEN, "Printed the first file", Style.RESET_ALL
#---------------------------------------------------------------------------------------------------------------
best_count = [greedy_best_count, gincex_best_count, k2means_best_count , tsp_best_count]
x          = np.arange(len(best_count))

fig1, ax1 = plt.subplots()
ax1.bar(x, best_count)
ax1.set_title("Histogram of number of times a heuristic performed the best." +\
              "\nNumber of Sites: " + str(num_pts_per_cloud)+ "      " +\
              r"$\varphi$: "     + str(phi)                         +\
              "\nCloudtype: " + cloud_type , fontsize=21)
plt.xticks(x,['greedy', 'gincex', 'k2means', 'tsp'], fontsize=15)
plt.yticks(range(0,max( best_count ) + 2, 3), fontsize=15)
plt.ylabel("Number of times this \nheuristic performed the best", fontsize=20)
plt.xlabel("Algorithm", fontsize=20)
plt.savefig('expt_comp_histo_mspan_nsites_' + str(num_pts_per_cloud)+\
            '_phi_'                   + str(phi) +\
            '_cloudtype_'             + cloud_type + '.png',\
            bbox_inches = 'tight', dpi=200)
print Fore.GREEN, "Printed the second file", Style.RESET_ALL
plt.show()
#---------------------------------------------------------------------------------------------------------------

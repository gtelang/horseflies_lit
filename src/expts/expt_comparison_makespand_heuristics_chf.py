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
                              
num_reps          = 40
num_pts_per_cloud = 5

mspans_greedy         = []
mspans_greedy_l1      = []
mspans_gincex         = []
mspans_gincoll        = []
mspans_gincl1         = []
mspans_k2means        = []
mspans_k2meansl1      = []
mspans_k3means        = []
mspans_k3meansl1      = []
mspans_tsp            = []
mspans_tspl1          = []

inithorseposn = np.asarray([0.5, 0.5])
phi           = 12.0

for i in range(num_reps):
    print "Rep: ", i
    sites = list(np.random.rand(num_pts_per_cloud,2))

    algo_greedy_data = chf.algo_greedy(sites, inithorseposn, phi, 
                                       write_algo_states_to_disk_p = False   ,
                                       animate_schedule_p          = False   , 
                                       post_optimizer              = chf.algo_exact_given_specific_ordering)

    mspans_greedy.append(algo_greedy_data['tour_length_with_waiting_time_included'])
    print Fore.CYAN, algo_greedy_data, Style.RESET_ALL
    #--------------------------------------------------------------------------------------------------------

    # algo_greedy_l1_data = chf.algo_greedy(sites, inithorseposn, phi, 
    #                                       write_algo_states_to_disk_p = False   ,
    #                                       animate_schedule_p          = False   , 
    #                                       post_optimizer              = chf.algo_approximate_L1_given_specific_ordering)

    # mspans_greedy_l1.append(algo_greedy_l1_data['tour_length_with_waiting_time_included'])
    # print Fore.CYAN, algo_greedy_l1_data, Style.RESET_ALL
    #--------------------------------------------------------------------------------------------------------
    algo_gincex_data = chf.algo_greedy_incremental_insertion(sites, inithorseposn, phi, 
                                                             write_algo_states_to_disk_p = False   ,
                                                             animate_schedule_p          = False   , 
                                                             post_optimizer              = chf.algo_exact_given_specific_ordering)

    mspans_gincex.append(algo_gincex_data['tour_length_with_waiting_time_included'])
    print Fore.CYAN, algo_gincex_data, Style.RESET_ALL
    #--------------------------------------------------------------------------------------------------------
    algo_gincoll_data = chf.algo_greedy_incremental_insertion(sites, inithorseposn, phi, 
                                                              write_algo_states_to_disk_p = False   ,
                                                              animate_schedule_p          = False   , 
                                                              post_optimizer              = None)

    mspans_gincoll.append(algo_gincoll_data['tour_length_with_waiting_time_included'])
    print Fore.CYAN, algo_gincoll_data , Style.RESET_ALL
    #--------------------------------------------------------------------------------------------------------

    # algo_gincl1_data = chf.algo_greedy_incremental_insertion(sites, inithorseposn, phi, 
    #                                                          write_algo_states_to_disk_p = False   ,
    #                                                          animate_schedule_p          = False   , 
    #                                                          post_optimizer              = chf.algo_approximate_L1_given_specific_ordering)
    # mspans_gincl1.append(algo_gincl1_data['tour_length_with_waiting_time_included'])
    # print Fore.CYAN, algo_gincl1_data, Style.RESET_ALL

    #--------------------------------------------------------------------------------------------------------
    algo_k2means_data = chf.algo_kmeans(sites, inithorseposn, phi, k=2,
                                        post_optimizer              = chf.algo_exact_given_specific_ordering)

    mspans_k2means.append(algo_k2means_data['tour_length_with_waiting_time_included'])
    print Fore.CYAN, algo_k2means_data, Style.RESET_ALL

    #--------------------------------------------------------------------------------------------------------
    # algo_k2meansl1_data = chf.algo_kmeans(sites, inithorseposn, phi, k=2,
    #                                       post_optimizer  = chf.algo_approximate_L1_given_specific_ordering)

    # mspans_k2meansl1.append(algo_k2meansl1_data['tour_length_with_waiting_time_included'])
    # print Fore.CYAN, algo_k2meansl1_data, Style.RESET_ALL

    #--------------------------------------------------------------------------------------------------------
    algo_k3means_data = chf.algo_kmeans(sites, inithorseposn, phi, k=3,
                                        post_optimizer              = chf.algo_exact_given_specific_ordering)

    mspans_k3means.append(algo_k3means_data['tour_length_with_waiting_time_included'])
    print Fore.CYAN, algo_k3means_data, Style.RESET_ALL

    #--------------------------------------------------------------------------------------------------------
    # algo_k3meansl1_data = chf.algo_kmeans(sites, inithorseposn, phi, k=3,
    #                                       post_optimizer  = chf.algo_approximate_L1_given_specific_ordering)

    # mspans_k3meansl1.append(algo_k3meansl1_data['tour_length_with_waiting_time_included'])
    # print Fore.CYAN, algo_k3meansl1_data, Style.RESET_ALL

    #--------------------------------------------------------------------------------------------------------
    algo_tsp_data = chf.algo_tsp_ordering_new(sites, inithorseposn, phi,
                                          post_optimizer  = chf.algo_exact_given_specific_ordering)
    mspans_tsp.append(algo_tsp_data['tour_length_with_waiting_time_included'])

    print Fore.CYAN, algo_tsp_data, Style.RESET_ALL

    #--------------------------------------------------------------------------------------------------------
    # algo_tspl1_data = chf.algo_tsp_ordering(sites, inithorseposn, phi,
    #                                         post_optimizer  =  chf.algo_approximate_L1_given_specific_ordering)

    # mspans_tspl1.append(algo_tspl1_data['tour_length_with_waiting_time_included'])
    # print Fore.CYAN, algo_tspl1_data, Style.RESET_ALL
    #--------------------------------------------------------------------------------------------------------


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots()
ax.set_xlabel("Runs", fontsize=25)
ax.set_ylabel("Makespans of Horsefly heuristics", fontsize=25)
ax.set_xticks(range(num_reps))
plt.grid(True, linestyle='--')
plt.tick_params(labelsize=25)

#ax.set_ylim([0.2,1.75])

plt.plot(range(num_reps) , mspans_greedy    , 'o-', label=r"greedy" ,color="blue")
plt.plot(range(num_reps) , mspans_gincex    , 'o-', label=r"gincex" ,color="orange")
#plt.plot(range(num_reps) , mspans_k2means   , 'o-', label=r"k2means",color="purple")
#plt.plot(range(num_reps) , mspans_k3means   , 'o-', label=r"k3means",color="red")
plt.plot(range(num_reps) , mspans_tsp       , 'o-', label=r"tsp"    ,color="green")


#plt.plot(range(num_reps) , mspans_gincoll   , 'o-', label=r"gincoll",color="olive")
#plt.plot(range(num_reps) , mspans_gincl1    , 'mo-', label=r"gincl1" ,color="brown")
#plt.plot(range(num_reps) , mspans_k2meansl1 , 'o-', label=r"k2meansl1",color="limegreen")
#plt.plot(range(num_reps) , mspans_k3meansl1 , 'o-', label=r"k3meansl1", color="tomato")
#plt.plot(range(num_reps) , mspans_greedy_l1 , 'go-', label=r"greedyl1")
#plt.plot(range(num_reps) , mspans_tspl1     , 'o-', label=r"tspl1"    , color="darkorange")

ax.set_title("Number of sites " + str(num_pts_per_cloud)+ "\n" + r"$\varphi$: " + str(phi) , fontsize=25)


ax.legend(loc='upper center', ncol=3, fancybox=True, shadow=True, prop={'size': 15})

plt.savefig('expt_comparison_makespan_numsites_'+str(num_pts_per_cloud)+ 'phi_'+str(phi) +'.png',dpi=300)
plt.show()

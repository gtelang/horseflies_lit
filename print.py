#---------------------------------
# Algorithms for reverse horsefly 
#---------------------------------
def algo_proceed_to_farthest_drone(sites, inithorseposn, phi,    \
                               write_algo_states_to_disk_p = False, \
                               write_io_p                  = False, \
                               animate_tour_p              = False,\
                               plot_tour_p                 = True) :
    def normalize(vec):
        assert(np.linalg.norm(vec)>1e-8) # you are not being passed the zero vector
        unit_vec =  1.0/np.linalg.norm(vec) * vec
        return unit_vec

    def furthest_uncollected_fly(current_horse_posn, uncollected_flies_idx):
        """ Find the furthest uncollected fly from the current horse position
        Replace this step with some dynamic farthest neighbor algorithm for 
        improving the speed."""
        imax = 0 
        dmax = -np.inf
        for idx in uncollected_flies_idx:
            dmax_test  = np.linalg.norm(sites[idx]-current_horse_posn)
            if dmax_test > dmax:
                imax = idx
                dmax = dmax_test
        return imax, dmax

    def calculate_interception_point(startpt, endpt, flypt):
        """ Consider the directed segment joining startpt and endpt with a truck beginning at `startpt'. 
        Consider a drone located at `flypt'. This function decides if a drone can intercept
        the truck as it travels from `startpt' to `flypt' at its full-speed assumed to be 1
        If not the function returns `None'. Otherwise it outputs the interception point. 
        """
        startpt, endpt     = map(np.asarray, [startpt, endpt])
        start_end_unit_vec = normalize(endpt-startpt)
        L                  = np.linalg.norm(endpt-startpt)

        alpha, beta = flypt
        coeffs      = [ 1.0 - phi**2  , -2.0 * alpha,  alpha**2 + beta**2  ]
        roots       = np.roots(coeffs)
        
        if all(np.isreal(roots)):
            if   0 <= roots[0] and roots[0] <= L  :
                return startpt + roots[0] * start_end_unit_vec
            elif 0 <= roots[1] and roots[1] <=L:
                return startpt + roots[1] * start_end_unit_vec 
            else:
                return None
        else:
            return None

    # Set algo-state and input-output files config
    import sys, datetime, os, errno

    if write_io_p:
        algo_name     = 'algo-proceed-to-farthest-drone'
        time_stamp    = datetime.datetime.now().strftime('Day-%Y-%m-%d_ClockTime-%H:%M:%S')
        dir_name      = algo_name + '---' + time_stamp
        io_file_name  = 'input_and_output.yml'

        try:
            os.makedirs(dir_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    numsites           = len(sites)
    numflies           = numsites
    current_horse_posn = np.asarray(inithorseposn) 

    horse_traj = [ {'coords'            : np.asarray(inithorseposn), 
                    'fly_idxs_picked_up': []                       , 
                    'waiting_time'      : 0.0} ] 

    flies_collected_p  = [False for i in range(numsites)] 
    order_collection   = []
    clock_time         = 0.0

    fly_trajs      = [[np.array(sites[i])] for i in range(numflies)]
    #---------------------------------------------------------------------------------
    # Main loop. Several flies will possibly be collected in each iteration unlike the 
    # Joe's original dead reckoning heuristic. 
    #---------------------------------------------------------------------------------
    while (not all(flies_collected_p)):
        current_horse_posn     = horse_traj[-1]['coords']
        uncollected_flies_idx  = [idx for idx in range(len(flies_collected_p)) if flies_collected_p[idx] == False]
        imax, dmax             = furthest_uncollected_fly(current_horse_posn, uncollected_flies_idx)

        # Extract position of farthest drone from current position of the truck 
        farthest_fly_posn    = sites[imax] 

        # Find the interception point with the furthest drone assuming ``head-on'' collision
        fly_idxs_picked_up = []
        for idx in uncollected_flies_idx:
            if idx != imax:
                maybe_meetpt = calculate_interception_point(current_horse_posn, farthest_fly_posn, sites[idx])
                if maybe_meetpt is not None:
                    fly_idxs_picked_up.append((idx, maybe_meetpt))
            else: 
                d      = np.linalg.norm(farthest_fly_posn-current_horse_posn)
                meetpt = current_horse_posn + d/(1.0+phi) * normalize(farthest_fly_posn-current_horse_posn)
                fly_idxs_picked_up.append((idx, meetpt))
         
        # Sort the interception points along the ray and add it to the horse trajectory
        fly_idxs_picked_up = sorted(fly_idxs_picked_up, \
                                    key=lambda (idx, meetpt): np.linalg.norm(meetpt-current_horse_posn))
        assert(fly_idxs_picked_up[-1][0] == imax)
        print Fore.GREEN
        utils_algo.print_list(fly_idxs_picked_up)
        print " ", Style.RESET_ALL
        for (idx, meetpt) in fly_idxs_picked_up:
            horse_traj.append({'coords'             : meetpt, 
                               'fly_idxs_picked_up' : [idx]        , 
                               'waiting_time'       : 0.0})

            fly_trajs[idx].append(meetpt)
            horse_traj_pts = [pt['coords'] for pt in horse_traj ]
            clock_time     = utils_algo.length_polygonal_chain(horse_traj_pts) 
            order_collection.append((idx, meetpt, clock_time)) 

            flies_collected_p[idx] = True


    if plot_tour_p:
        print "------------------Final Horse Traj------------"
        utils_algo.print_list(horse_traj)
        from   matplotlib.patches import Circle
        import matplotlib.pyplot as plt 

        # Set up configurations and parameters for all necessary graphics
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        sitesize = 0.010
        fig, ax = plt.subplots()
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_aspect('equal')

        ax.set_xticks(np.arange(0, 1, 0.1))
        ax.set_yticks(np.arange(0, 1, 0.1))

        # Turn on the minor TICKS, which are required for the minor GRID
        ax.minorticks_on()

        # Customize the major grid
        ax.grid(which='major', linestyle='--', linewidth='0.3', color='red')

        # Customize the minor grid
        ax.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

        # Plot Fly segments
        for ftraj  in fly_trajs:
            xfs = [ftraj[0][0], ftraj[-1][0]]
            yfs = [ftraj[0][1], ftraj[-1][1]]
            ax.plot(xfs,yfs,'-',linewidth=2.0, markersize=3, alpha=0.7, color='g')

        # Plot sites as small disks (these are obviosuly the initial positions of the flies)
        for site in sites:
            circle    = Circle((site[0], site[1]), sitesize, facecolor='k', edgecolor='black',linewidth=1.0)
            sitepatch = ax.add_patch(circle)

        # Plot initial position of the horse 
        circle = Circle((inithorseposn[0], inithorseposn[1]), 0.02, \
                        facecolor = '#D13131', edgecolor='black', linewidth=1.0)
        ax.add_patch(circle)
        
        # Plot Horse tour
        xhs = [ pt['coords'][0] for pt in horse_traj ]
        yhs = [ pt['coords'][1] for pt in horse_traj ]
        ax.plot(xhs,yhs,'-',linewidth=5.0, markersize=6, alpha=1.00, color='#D13131')

        # Plot meta-data
        tour_length = utils_algo.length_polygonal_chain(zip(xhs, yhs))
        ax.set_title("Algo: fh \n Number of sites: " + str(len(sites)) + "\n Tour length " +\
                    str(round(tour_length,4)), fontsize=25)
        ax.set_xlabel(r"$\varphi=$ " + str(phi) , fontsize=25)
        plt.show()





    return horse_traj, fly_trajs




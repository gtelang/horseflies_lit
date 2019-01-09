
# Turn off Matplotlibs irritating DEBUG messages
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

# Import problem module files
   
import sys
sys.path.append('src/lib')
import problem_classic_horsefly as chf
import problem_one_horse_multiple_flies as ohmf


if __name__=="__main__":
     # Select algorithm or experiment 
     if (len(sys.argv)==1):
          print "Specify the problem or experiment you want to run"
          sys.exit()

     elif sys.argv[1] == "--problem-classic-horsefly":
          chf.run_handler()

     elif sys.argv[1] == "--problem-one-horse-multiple-flies":
          ohmf.run_handler()

     else:
          print "Option not recognized"
          sys.exit()

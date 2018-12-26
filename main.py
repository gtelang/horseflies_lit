
# Turn off Matplotlib's irritating DEBUG messages
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

# Import problem module files
   
import sys
sys.path.append('src/lib')
import problem_classic_horsefly as chf
#import problem_segment_horsefly as shf
#import problem_one_horse_two_flies as oh2f


if __name__=="__main__":
     # Select algorithm or experiment 
     if (len(sys.argv)==1):
          print "Specify the problem or experiment you want to run"
          sys.exit()

     elif sys.argv[1] == "--problem-classic-horsefly":
          chf.run_handler()

     elif sys.argv[1] == "--problem-segment-horsefly":
          shf.run_handler()

     elif sys.argv[1] == "--problem-one-horse-two-flies":
          oh2f.run_handler()

     else:
          print "Option not recognized"
          sys.exit()

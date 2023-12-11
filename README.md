# CSC746 Project: Parallelized Genome Sequencing and Bioinformatics

main.cu: contains CUDA/GPU implementation \\

cpu_main.cpp: contains OpenMP/CPU based implementation \\

run.sh: A reference bash script for running programs through varying problem sizes \\

sequences.rtf: Contains sample sequences of varying sizes for pasting into the code and running \\


# How to run

## main.cu 
Compile the script: nvcc main.cu -o smith_waterman \\

Obtain performance metrics: ncu --set default --section SourceCounters --metrics smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --target-processes all ./smith_waterman 1 1 --kernel-id smithWatermanKernel \\

where the "1 1" before kernel ID is the concurrency, it may be change to: 2 2, 4 4 and so on

## cpu_main.cpp
Compile and run the script: \\

g++ -fopenmp cpu_main.cpp -o sw_cpu \\

chmod +x sw_cpu \\

srun ./sw_cpu 1 \\

where 1 runs the program on 1x1, 2 runs the program on 2x2 and so on \\

cpu_main.cpp: make sure you change line number 14 based on instructions in line number 13, depending on the problem size you are running.

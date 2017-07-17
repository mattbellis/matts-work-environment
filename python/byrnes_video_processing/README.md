The folder "JointAnglesFiles" contains numerous joint angles that I've calculated for each trial.

The "ja_labels" file contains label names for each column in the "...xyz_ja.txt" files.

The other 2 files contain rows that include the filename and start and end times for each stride.

There is one additional column (#2) in those files for housekeeping which says which frame of the movie the 
first digitized point for each trial is.

I also just remembered that python starts it's numbering at 0, not 1, so we'll have to subtract 
1 frame from the start/end times.

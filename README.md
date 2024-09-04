# README 

## Project Overview:
Pain is a salient driver of change in our daily lives. We move differently, avoid activities, and often experience lower quality of life as a result. This project combines several areas, including psychophysics, neuromechanics, and real-time feedback systems to understand how and why we move when experiencing pain.

## Initializations   
1. download repo
2. create fresh virtual environment. Conda was used for this project - use another environment manager at your own discretion
3. install packages using requirements.txt (e.g., conda install --file requirements.txt)
4. Now you should be able to run the different scripts as you desire 

## Experiment 1
Data and results are in "./experiment_one/"
### Analysis 1 
Compare differences in knee joint angle values averaged over the first 10 strides and last ten strides of the experimental pain conditions with respect to baseline walking.    
*Note*   The reference frame for angles was such that more negative angles equate to more knee flexion.   

Outcomes:  
- peak_KSA_stance_mean = Minimum (peak) knee angle in stance  
- peak_KSA_swing_mean  = Minimum (peak) knee angle in swing
- min_KSA_stance_mean  = Maximum knee angle in midstance

Data:
- summary_discrete_first_ten.pkl = discrete knee angle variables from the first ten strides of each trial
- summary_discrete_last_ten.pkl = discrete knee angle variables from the last ten strides of each trial

Statistical Comparisons:
- Run discrete_knee_biomech.py
- This will produce 6 files
  - discrete_biomech_wilcox_[first/last]_ten = statistical results
  - discrete_biomech_difference_summary_[first/last]_ten = difference values from baseline to each trial
  - discrete_biomech_summary_[first/last]_ten = summary statistics for each trial

Visualizations:  
biomech_vis.py
- Loads representative participant data, normalized knee angle waveforms, and discrete data
- generates figures using in manuscript to visualize knee angles
- produces 3 .svg plots
  - discrete_knee_angle_3panel.svg = shows each variable using a 10-stride moving window over the duration of the 10-minute walking bouts
  - mean_waveforms_first_ten.svg = normalized knee angles from the first ten strides of each trial, averaged across participants
  - representative_partic_knee_angle = raw knee angle data from one participant's baseline and trial 1 data to show what original data looks like and identify discrete outcomes (see figure in manuscript).

### Analysis 2   
Compare pain intensity ratings within and between three walking trials each lasting 600 seconds.   

Data:
- data.xlsx sheet = painratings inlcudes each participant's pain ratings for each of the 21 time points.
- pain_time_data.pkl = a compiled version of participant's pain ratings.   
- habituation_fit_results.pkl = exponential modelling of pain-time data

Statistical Comparisons:
- run locohab_pain.py
- this will produce 3 files
  - painrating_wilcox.xlsx = contains stastical tests for each trial (t1 ,t2, t3) for each timepoint compared to the initial pain rating
  - painrating_summary_statistics.xlsx = summary statistics for each trial and timepoint across participants
  - initialpain_comparison.xlsx = statistical comparisons of the initial pain ratings at the start of trial 1, trial 2, and trial 3 (t1, t2, t3).

Visualizations:  
locohab_pain_batch.py
- pain_habituation_plot-3panel.svg = a single 3-panel figure with individual participant pain trajectories over the 3 10-minute walking bouts

### Analysis 3
Compare changes in the area and location of the pain regions illustrated by particpants after each walking trial

Data:
- all_maps_frontal.pkl = array of 1875x1875 binary images across participants for the frontal knee view.
- all_maps_transverse.pkl = array of 1875x1875 binary images across participants for the transverse knee view.
- all_maps_moments.xlsx = the variables extracted from the images using OpenCV
  - x = centroid location in x direction
  - y = centroid location in y direction
  - area_px = area of pixels coloured by participants relative to the whole image.
  - area_perc = percent area (area_px / total number of pixels)
  - contour_number = nominal index of contour on a given image
  - fname = kneediag_[ntrial number] indicating the walking trial the image is associated with.
  - cond = same as above but represented as an integer
  - view = Page 1 or Page 2, referencing frontal and transverse views respectively
  - participant = participant code
  
Statistical Comparison:
- run kneemap_area_pos.py and change variable 'proc_data_path' to "experiment_one"
- this will produce 3 files
  - painregions_area_statistics = statistical comparisons between each trial for area anc centroid locations
  - painregions_area_changedata = change and and percent change of the area variable between trials
  - painregions_summary_data = summary statistics for area and contour variables for each trial

Visualizations:  
kneemaps_figures.py
- represents the regions drawn by participants as a heatmap with a centroid position overlaid.
- generates 6 .svg files "kneemap_" + "fr, tr" + "1,2,3" where fr/tr represent frontal or transverse views and 1,2,3 are the trials


## Experiment 2
### Analysis 1 
Model the relationship between pain intensity as a function of electrical stimulation magnitude (milliamperes; mA).

Data:
- pain_current_fits.pkl = all the individual stimuli-pain paired values and the modelling results for a linear, exponential, and piecewise linear model.

Visualizations:   
painmap_modelling.py
- produces one figure that shows two representative participants and all participant data. Linear, exponential, and piecewise linear models based on the median coefficients across participants are included on the figures.
- adjusted coefficients of determination are included to quantify model performance. 


### Analysis 2
Visualize within-gait cycle modulations in pain during slow (25 stride/minute) locomotion

Data:
- pain_traces_50spm.pkl = all summary and raw data for each participant stored in a dictionary.

Visualizations:   
pain_gaitcycle.py
- generates a 3 panel figure (pain_modulator_3panel.svg) that shows
  - two reperesenative participants pain intensity data over the 75 strides of walking
  - the ensemble averages of pain intensity normalized to gait cycle for these two representative participants. *note* an inset axis is also generated (pain_modulator_inset.svg), see figure in manuscript
  - a heatmap showing the average pain intensity across participants and the gait cycle, for each of the 75 strides.

### Analysis 3
This is the exact same process as is outlined in Experiment 1-Analysis 3. The only difference is the regional maps are for pain intensities of 1/10, 3/10 and 5/10 (on a 0-10 numerical rating scale)

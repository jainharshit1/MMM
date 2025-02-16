# MMM_media_mix_modeling
#Project Overview
This is the repositry that contains my ongoing project on media mix modelling, an ML model.which I started during my 2024-25 summer intern at Growth Natives.
# What is MMM?
Before explaining the work-flow of the project I would like to highlight the use of the project, MMM is an ML model that has a real-world use in the Digital Marketing domain. This model's training is run on the data provided by the companies who wish to optimize their Marketing-Spend, without having to rely on the user based inputs.
#Flow of the project
Call_model :-this is the main file of the project that is used to call all other contributing files. the first step of the code is data-cleaning. As the data collected is completely raw to un an ML model on it , data cleaning is required.This step is carried out with the help of the data-prep file.
Carryover & Saturation files (in the ChannelContrib folder):- The next step of the code is to add the carryover and saturation effects to the spend of the clients across various channels. This is necessary as the data being time series one we can't negelect the adstock effect.
Hyper-Paramete-Tuning:- while selecting the best functions and their suitable parameters to capture the carryover and the saturation effects of the data we used the optuna library. Which tuned all the parameters adequately on the basis of least MSE.
After deciding the transformations the data is passed through a pipeline where the feature scaling and the transformations are applied.
Then the Ridge regression is fitted onto the model, which although being only moderately accurate is a necessity because of the easy interpretability of it's coefficients.
The results are then displayed through charts.
The project is still an ongoing one, as there is still a lot of experimentation left, like the correct addition of the lag effect and the interaction effect of the features. 
The future plan is to add more accurate functions to capture the adstock effect and to experiment with the bayesian regression techniques as currently only the traditional ML techniques have been tested. 

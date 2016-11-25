# PMRF
Partially Monotone Random Forest

This repository provides all PYTHON code necessary to use build Partially Monotone Random Forest models for binary classification problems. The framework is described in Bartey, Liu and Reynolds 'A Novel Technique for Integrating Monotone Domain Knowledge into the Random Forest Classifier' 2016 (accepted by AusDM 2016). Please see the paper for further details.

A typical process to build a partially monotone knowledge integrated model is:

1. Determine optimal 'maximum_features' (mtry) hyperparameter by parameter sweep on e.g. {1,2,3,5,7...} and using optimal Out of Box (OOB) accuracy estimate for standard RF (use sklearn RandomForestClassifier).

2. Solve PMRandomForestClassifier on all training data and with NO constraints.

3. You can assess monotonicity of the unconstrained model for particular features using calc_mcc_interp in the PM repository (just wrap the pmrf model in model_pmrf_c()).

4. Decide which features are suggested monotone in the response, based on domain knowledge.

5. Build a constraint set using 'gen_constrset_adaptive' in the PM repository (\chriswbartley\PM). Pass in the trained PMRF model wrapped in the model_pmrf_c() object (just a wrapper so that the predict() function is clearly mapped).

6. Solve the CONSTRAINED RF using pmrf.fitConstrained() and passing in the generated constraints. 

7. Once trained, use predictConstrained() to predict using the monotone model.

8. You can assess resulting monotonicity of the model for particular features using calc_mcc_interp in the PM repository (just wrap the pmrf model in model_pmrf_c()).


DEPENDENCIES:
The constraint generation (gen_constrset_adaptive()) and montonicity measurement code (calc_mcc_interp) is in the separate 'PM' repository as it is generic to other kinds of models.


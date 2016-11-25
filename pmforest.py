# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:50:36 2016

@author: Hi
"""

#from __future__ import division
#
#import warnings
#from warnings import warn
#
#from abc import ABCMeta, abstractmethod
#
#import numpy as np
#from scipy.sparse import issparse
#
#from sklearn.base import ClassifierMixin, RegressorMixin
#from sklearn.externals.joblib import Parallel, delayed
#from sklearn.externals import six
#from sklearn.feature_selection.from_model import _LearntSelectorMixin
#from sklearn.metrics import r2_score
#from sklearn.preprocessing import OneHotEncoder
from warnings import warn
from sklearn.tree import (DecisionTreeClassifier)#, DecisionTreeRegressor,
                  #  ExtraTreeClassifier, ExtraTreeRegressor)
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, check_array
#from sklearn.tree._tree import DTYPE, DOUBLE
#from sklearn.utils import check_random_state, check_array, compute_sample_weight
#from sklearn.utils.validation import DataConversionWarning, NotFittedError
#from scipy.base import BaseEnsemble, _partition_estimators
#from sklearn.utils.fixes import bincount
#from sklearn.utils.multiclass import check_classification_targets
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.fixes import bincount
import numpy as np
from quadprog import solve_qp
import copy
import sympy
#from sklearn.ensemble import ForestClassifier
__all__ = ["PMRandomForestClassifier"]

MAX_INT = np.iinfo(np.int32).max

def unique_rows(A, return_index=False, return_inverse=False):
    """
    Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]

    Returns I if return_index is True
    Returns J if return_inverse is True
    """
    A = np.require(A, requirements='C')
    assert A.ndim == 2, "array must be 2-dim'l"

    B = np.unique(A.view([('', A.dtype)]*A.shape[1]),
               return_index=return_index,
               return_inverse=return_inverse)

    if return_index or return_inverse:
        return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
            + B[1:]
    else:
        return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')
 
def check_constraints(leaves_lower,leaves_higher):
    # check for constraints with same leaves for all trees
    comparison=  leaves_lower== leaves_higher
    comparison=np.sum(comparison,1)
    comparison=comparison!=leaves_lower.shape[1] # gives bool mask for good constraints
    indexes_good1=np.arange(leaves_lower.shape[0])[comparison]
    #check for duplicate constraints
    constr_dupchk=np.hstack((leaves_lower[indexes_good1,:],leaves_higher[indexes_good1,:]))
    uniq_indexes=unique_rows(constr_dupchk, return_index=True, return_inverse=False)
    uniq_indexes=uniq_indexes[1]
    return indexes_good1[uniq_indexes]       
def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices

def _generate_unsampled_indices(random_state, n_samples):
    """Private function used to forest._set_oob_score fuction."""
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices


class PMRandomForestClassifier():
    """A random forest classifier with PARTIAL MONOTONICITY support.
    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and use averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).
    Read more in the :ref:`User Guide <forest>`.
    Parameters
    ----------
    proximity_mode: 'training_samples' or 'tree_samples' (default). 'training_samples'  
        means all training points are put down all trees.'tree_sample' uses only the 
        samples used to estimate each tree (default random forest formuation)
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.
    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
        Note: this parameter is tree-specific.
    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.
        Note: this parameter is tree-specific.
    min_samples_split : integer, optional (default=2)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.
    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples in newly created leaves.  A split is
        discarded if after the split, one of the leaves would contain less then
        ``min_samples_leaf`` samples.
        Note: this parameter is tree-specific.
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.
        Note: this parameter is tree-specific.
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.
        Note: this parameter is tree-specific.
    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.
    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or None, optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that weights are
        computed based on the bootstrap sample for every tree grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.
    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).
    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.
    constraints : a [MxPx2] array of constraints used in the solution
    References
    ----------
    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
    See also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier
    """
    def __init__(self,base_rfc=None,train_X=None,train_Y=None,pm_weights=None,proximity_mode='tree_samples',oob_score_constrained=False,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        if base_rfc==None:
            self.rfc=RandomForestClassifier(
                n_estimators=n_estimators,
                criterion = criterion,
                max_depth = max_depth,
                min_samples_split = min_samples_split,
                min_samples_leaf = min_samples_leaf,
                min_weight_fraction_leaf = min_weight_fraction_leaf,
                max_features = max_features,
                max_leaf_nodes = max_leaf_nodes,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight)
        else:
            self.rfc=base_rfc
        self.train_X=train_X
        self.train_Y=train_Y
        self.__proximity_mode=proximity_mode
        self.__pm_weights=pm_weights
        self.__training_leafnodes=None
        self.__training_leafnodes_sparse=None
        self.__training_leaf_populations=None
        self.__leaf_predictions=None
        self.__leaf_predictions_unc =None
        self.oob_score_constrained=oob_score_constrained
        self.oob_score_constrained_=None
    @property #get
    def pm_weights(self):
        if self.__pm_weights is None:
            numsamples=len(self.train_X) 
            return np.zeros(numsamples)+1.0/np.double(numsamples) 
        else:
            return self.__pm_weights   

    @pm_weights.setter
    def pm_weights(self, x):
        self.__pm_weights = x
        
    @property #get
    def leaf_predictions(self):
        if self.__leaf_predictions is None:
            if not self.train_X is None:
                numsamples=len(self.train_X) #.shape[0] #1 if len(testX.shape)==1 else testX.shape[0]
                p=self.pm_weights
                arrtrainY=np.asarray(self.train_Y).reshape(len(self.train_Y))
                max_num_leaves=np.max(self.training_leafnodes)+1
                num_trees=self.rfc.get_params()['n_estimators']
                tmp=np.zeros((max_num_leaves,num_trees))
                for jtree in np.arange(0,np.int(num_trees)):
                    for i in np.arange(0,numsamples):
                        ileaf=self.training_leafnodes[i,jtree]
                        if ileaf!=-99.: # marker for 'not in this tree' (proximity_mode='tree_samples' only)
                            tmp[ileaf,jtree]=tmp[ileaf,jtree]+p[i]*arrtrainY[i]*1.0*self.training_sample_counts[i,jtree]/np.double(num_trees)/self.training_leaf_populations[jtree][ileaf]
                self.leaf_predictions=tmp  
        return self.__leaf_predictions   

    @leaf_predictions.setter
    def leaf_predictions(self, x):
        self.__leaf_predictions = x
       
    @property #get
    def leaf_predictions_unc(self):
        if self.__leaf_predictions_unc is None:
            if not self.train_X is None:
                numsamples=len(self.train_X) #.shape[0] #1 if len(testX.shape)==1 else testX.shape[0]
                p=np.zeros(self.num_samples)+1.0/self.num_samples
                arrtrainY=np.asarray(self.train_Y).reshape(len(self.train_Y))
                max_num_leaves=np.max(self.training_leafnodes)+1
                num_trees=self.rfc.get_params()['n_estimators']
                tmp=np.zeros((max_num_leaves,num_trees))
                for jtree in np.arange(0,np.int(num_trees)):
                    for i in np.arange(0,numsamples):
                        ileaf=self.training_leafnodes[i,jtree]
                        if ileaf!=-99.: # marker for 'not in this tree' (proximity_mode='tree_samples' only)
                            tmp[ileaf,jtree]=tmp[ileaf,jtree]+p[i]*arrtrainY[i]*1.0*self.training_sample_counts[i,jtree]/np.double(num_trees)/self.training_leaf_populations[jtree][ileaf]
                self.leaf_predictions_unc=tmp  
        return self.__leaf_predictions_unc   

    @leaf_predictions_unc.setter
    def leaf_predictions_unc(self, x):
        self.__leaf_predictions_unc = x
        
    @property #get
    def training_leafnodes(self):
        if self.__training_leafnodes is None:
            if not self.train_X is None:
                if self.__proximity_mode=='training_samples':
                    self.training_leafnodes=self.rfc.apply(self.train_X)
                    self.training_sample_counts=np.ones(self.training_leafnodes.shape,dtype='double')
                elif self.__proximity_mode=='tree_samples': 
                    tmp=self.rfc.apply(self.train_X)
                    sample_counts=np.zeros(tmp.shape,dtype='double')
                    # remove leaf nodes for trees that each sample was NOT used to train
                    trees=self.rfc.estimators_
                    for j in np.arange(self.num_trees):
                        tree=trees[j]
                        indexes_of_samples_used_for_this_tree=_generate_sample_indices(tree.random_state, self.num_samples)
                        indexes_of_samples_not_used_for_this_tree=_generate_unsampled_indices(tree.random_state, self.num_samples)
                        tmp[indexes_of_samples_not_used_for_this_tree,j]=-99. # mark as not present in this tree
                        sample_counts[:,j]= bincount(indexes_of_samples_used_for_this_tree, minlength=self.num_samples)
                    self.training_leafnodes=tmp  
                    self.training_sample_counts=sample_counts
        return self.__training_leafnodes   
    
    @training_leafnodes.setter
    def training_leafnodes(self, x):
        self.__training_leafnodes = x
        
    @property #get
    def training_sample_counts(self):
        f=self.training_leafnodes # this getter ensures  __training_sample_counts is set appropriately  
        return self.__training_sample_counts  
        
    @training_sample_counts.setter
    def training_sample_counts(self, x):
        self.__training_sample_counts = x
     

    
        
    @property #get
    def training_leaf_populations(self):
        if self.__training_leaf_populations is None:
            if not self.train_X is None:
#                if self.__proximity_mode=='training_samples':
#                    leafpopulations={}
#                    for itree in np.arange(self.num_trees):
#                        popn=np.zeros(np.int(np.max(self.training_leafnodes[:,itree])),dtype=int)
#                        for inode in np.arange(np.int(np.max(self.training_leafnodes[:,itree]))):
#                            popn[inode]=np.sum([1 for leafnode in self.training_leafnodes[:,itree] if leafnode==(inode+1)])
#                        leafpopulations[itree]=popn  # 40: test accuracy (BREIMAN UNIT VOTE)
#                    self.__training_leaf_populations=leafpopulations
#                elif self.__proximity_mode=='tree_samples': 
                leafpopulations={}
                for itree in np.arange(self.num_trees):
                    popn=np.zeros(np.int(np.max(self.training_leafnodes[:,itree])+1),dtype=int)
                    for inode in np.arange(np.int(np.max(self.training_leafnodes[:,itree]))+1):
                        popn[inode]=np.sum([count for leafnode,count in zip(self.training_leafnodes[:,itree],self.training_sample_counts[:,itree]) if leafnode==(inode)])
                    leafpopulations[itree]=popn  # 40: test accuracy (BREIMAN UNIT VOTE)
                self.__training_leaf_populations=leafpopulations
        return self.__training_leaf_populations   

    @training_leaf_populations.setter
    def training_leaf_populations(self, x):
        self.__training_leaf_populations = x
        
    @property #get
    def num_trees(self):
        return self.rfc.get_params()['n_estimators']
    
    @property #get
    def num_samples(self):
        return len(self.train_X) 
    @property #get
    def training_leafnodes_sparse(self):
        if self.__training_leafnodes_sparse is None:
            if not self.train_X is None:
                max_num_leaves=np.max(self.training_leafnodes)
                n=len(self.train_X)
                num_trees=self.rfc.get_params()['n_estimators']
                p=self.pm_weights
                tmp=np.zeros((n,max_num_leaves,num_trees),dtype='float')
                for i_x in np.arange(n):
                    for j_tree in np.arange(num_trees):
                        tmp[i_x,self.training_leafnodes[i_x,j_tree]-1,j_tree]=np.double(self.train_Y[i_x])*p[i_x]/self.training_leaf_populations[j_tree][self.training_leafnodes[i_x,j_tree]] #removed: -1
                self.__training_leafnodes_sparse=tmp
        return self.__training_leafnodes_sparse   

    @training_leafnodes_sparse.setter
    def training_leafnodes_sparse(self, x):
        self.__training_leafnodes_sparse = x    
        #self.constraints=np.zeros(0);
    
    
    def convert_leafnodes_to_sparse(self,leafnodes,numleafnodes):
        max_num_leaves=numleafnodes #np.max(leafnodes)
        n=leafnodes.shape[0]
        num_trees=leafnodes.shape[1]
        tmp=np.zeros((n,max_num_leaves,num_trees),dtype='float')
        for i_x in np.arange(n):
            for j_tree in np.arange(num_trees):
                tmp[i_x,leafnodes[i_x,j_tree]-1,j_tree]=1.
        return tmp
    def fit(self, X, y, sample_weight=None):
        
        self.train_X=X
        self.train_Y=y
        self.__pm_weights=None
        self.__training_leaf_populations=None
        self.__training_leafnodes=None
        self.__training_leafnodes_sparse=None
        self.__leaf_predictions=None
        self.__leaf_predictions_unc=None
        res= self.rfc.fit(X, y, sample_weight)
        if self.oob_score_constrained:
            self._set_oob_score_constrained(X,y)
        return res
        
#    def getLeafPopulations(self,leafnodedata):
#        numTrees=leafnodedata.shape[1]
#        leafpopulations={}
#        for itree in np.arange(numTrees):
#            popn=np.zeros(np.int(np.max(leafnodedata[:,itree])),dtype=int)
#            for inode in np.arange(np.int(np.max(leafnodedata[:,itree]))):
#                popn[inode]=np.sum([1 for leafnode in leafnodedata[:,itree] if leafnode==(inode+1)])
#            leafpopulations[itree]=popn  # 40: test accuracy (BREIMAN UNIT VOTE)
#        return leafpopulations
        
    def predictConstrained(self,testX,useconstraints=True,use_trees='all'): 
        if testX.shape[0]==0:
            return np.array([])
        else:
            classes= np.sign(self.predictConstrainedReal(testX,useconstraints))
            classes[classes==0.]=-1. # choose a definite class: default to -1 as this matches std scikit rf
            return classes if len(classes)>1 else classes[0]
#    def predictConstrainedReal2(self,testX,useconstraints=True):  
#        if len(testX.shape)==1:
#            testX=testX.reshape(1,-1) # ensure 2D array
#        sparse_train=self.training_leafnodes_sparse
#        sparse_test=self.convert_leafnodes_to_sparse(self.rfc.apply(testX),sparse_train.shape[1])
#        numTrees=np.int(self.rfc.get_params()['n_estimators'])
#        n=len(self.train_X)
#
#        # use kNN approach to predict RF class
#        numsamples=1 if len(testX.shape)==1 else testX.shape[0] #1 if len(testX.shape)==1 else testX.shape[0]
#        results=np.zeros((numsamples,n,numTrees),dtype='float')
#        for jtree in np.arange(0,np.int(numTrees)):
#            #print(sparse_test[:,:,jtree].shape)
#            #print(sparse_train[:,:,jtree].T.shape)
#            results[:,:,jtree]=np.dot(sparse_test[:,:,jtree],sparse_train[:,:,jtree].T) #sparse_test[:,:,jtree]*sparse_train[:,:,jtree].T
#        results=np.sum(results,axis=1)
#        results=np.sum(results,axis=1)
##        for ksample in np.arange(numsamples):
##            res=0.0
##            test_leafnodes=test_leafnodes_all[ksample,:]
##            for i in np.arange(0,n):
##                for jtree in np.arange(0,np.int(numTrees)):
##                    thistree_testleafnode=test_leafnodes[jtree]
##                    thistree_trainptleafnode=self.training_leafnodes[i,jtree]
##                    if thistree_testleafnode==thistree_trainptleafnode:
##                        res=res+p[i]*arrtrainY[i]*1.0/np.double(numTrees)/self.training_leaf_populations[jtree][thistree_testleafnode-1]
##            results[ksample]=np.double(n)*res
#        results= results/np.double(numTrees)*np.double(n)
#        return results #if len(results)>1 else results[0]

            
    def predictConstrainedReal(self,testX,useconstraints=True,use_trees='all'):  
        if testX.shape[0]==0:
            return np.array([])
        else:
            leafpreds=self.leaf_predictions if useconstraints else self.leaf_predictions_unc
            numTrees=np.int(self.rfc.get_params()['n_estimators'])
            trees= np.arange(0,np.int(numTrees)) if use_trees=='all' else use_trees
            n=len(self.train_X)
            if len(testX.shape)==1:
                testX=testX.reshape(1,-1) # ensure 2D array
            test_leafnodes_all=self.rfc.apply(testX)
            # use kNN approach to predict RF class
            
            numsamples=testX.shape[0] #1 if len(testX.shape)==1 else testX.shape[0]
            results=np.zeros(numsamples)
            
            for ksample in np.arange(numsamples):
                res=0.0
                test_leafnodes=test_leafnodes_all[ksample,:]
                for jtree in trees:
                    res=res+leafpreds[test_leafnodes[jtree],jtree]
                results[ksample]=np.double(n)*res
            return results     
        
#    def predictConstrainedReal(self,testX,useconstraints=True):  
#        numTrees=np.int(self.rfc.get_params()['n_estimators'])
#        arrtrainY=np.asarray(self.train_Y).reshape(len(self.train_Y))
#        n=len(self.train_X)
#        if useconstraints and not(self.__pm_weights is None):
#            p=self.pm_weights
#        else:
#            p=np.zeros(n)+1.0/np.double(n)
#        if len(testX.shape)==1:
#            testX=testX.reshape(1,-1) # ensure 2D array
#        test_leafnodes_all=self.rfc.apply(testX)
#        # use kNN approach to predict RF class
#        
#        numsamples=testX.shape[0] #1 if len(testX.shape)==1 else testX.shape[0]
#        results=np.zeros(numsamples)
#        
#        for ksample in np.arange(numsamples):
#            res=0.0
#            test_leafnodes=test_leafnodes_all[ksample,:]
#            for i in np.arange(0,n):
#                for jtree in np.arange(0,np.int(numTrees)):
#                    thistree_testleafnode=test_leafnodes[jtree]
#                    thistree_trainptleafnode=self.training_leafnodes[i,jtree]
#                    if thistree_testleafnode==thistree_trainptleafnode:
#                        res=res+p[i]*arrtrainY[i]*1.0/np.double(numTrees)/self.training_leaf_populations[jtree][thistree_testleafnode-1]
#            results[ksample]=np.double(n)*res
#        return results


        
    def fitConstrained(self,constraints_lower_arg,constraints_higher_arg):
        self.__leaf_predictions=None
        numtrees=np.int(self.rfc.get_params()['n_estimators'])
        if len(constraints_higher_arg)==0:
            self.__pm_weights =None
        else:
            # see if any constraints are in the same leaf nodes for all trees
            test_leafnodes_higher_all_test=self.rfc.apply(constraints_higher_arg)
            test_leafnodes_lower_all_test=self.rfc.apply(constraints_lower_arg)
            # check constraits for no change between higher /lower, and duplicates:
            good_constr_indexes=check_constraints(test_leafnodes_lower_all_test,test_leafnodes_higher_all_test)
            constraints_higher=constraints_higher_arg[good_constr_indexes,:].copy()
            constraints_lower=constraints_lower_arg[good_constr_indexes,:].copy()
            test_leafnodes_higher_all=test_leafnodes_higher_all_test[good_constr_indexes,:].copy()
            test_leafnodes_lower_all=test_leafnodes_lower_all_test[good_constr_indexes,:].copy()
            if constraints_higher_arg.shape[0]>constraints_higher.shape[0]:
                print('Duplicates and/or stationary constraints found. Orig: ' + str(constraints_higher_arg.shape[0])  + ' Reduced: ' + str(constraints_higher.shape[0]))
            numConstr=len(constraints_lower)
            if numConstr==0:
                self.__pm_weights =None
            else: #numConstr>0 - ie we hve some constraints!
                # Build QP matrices
                #Minimize     1/2 x^T G x - a^T x
                #Subject to   C.T x >= b    
                n=len(self.train_X)
                G=np.eye(n)
                a=np.zeros(n)+1.0/np.double(n)
                C0=np.zeros((1,n))+1.0
                C1=np.zeros((numConstr,n))
                b=np.zeros((numConstr+1))
                b[0]=1.0
                
                #self.leafpopulations=self.getLeafPopulations(super(PMRandomForestClassifier, self).apply(self.train_X))
                
                train_Y_arr=np.double(np.ravel(self.train_Y))
                for kconstr in np.arange(numConstr):
                    test_leafnodes_higher=test_leafnodes_higher_all[kconstr,:]
                    test_leafnodes_lower=test_leafnodes_lower_all[kconstr,:]
                    for i in np.arange(0,len(self.train_X)):
                        res=0.0
                        for jtree in np.arange(0,np.int(numtrees)):
                            thistree_trainptleafnode=self.training_leafnodes[i,jtree]
                            thistree_trainptleafnode_count=self.training_sample_counts[i,jtree]
                            # do higher constraint value with PLUS
                            thistree_testleafnode_higher=test_leafnodes_higher[jtree]
                            if thistree_testleafnode_higher==thistree_trainptleafnode:
                                res=res+train_Y_arr[i]*1.0*thistree_trainptleafnode_count/np.double(numtrees)/np.double(self.training_leaf_populations[jtree][thistree_testleafnode_higher])#removed: -1
                            # do lower constraint value with MINUS
                            thistree_testleafnode_lower=test_leafnodes_lower[jtree]
                            if thistree_testleafnode_lower==thistree_trainptleafnode:
                                res=res-train_Y_arr[i]*1.0*thistree_trainptleafnode_count/np.double(numtrees)/np.double(self.training_leaf_populations[jtree][thistree_testleafnode_lower])#removed: -1
                            if np.isnan(res):
                                print('sdfsdf')
                        C1[kconstr,i]=res
                C=np.zeros((numConstr+1,n))
                C[0,:]=C0
                C[1:,:]=C1
                # Solve QP for new sample weights
                p=solve_qp(G,a,C.T,b,1,False)
                self.pm_weights= p[0] # extract coefficients
                #print(p[0][p[0]<0.99*(1.0/np.double(n))])
                #return 0
        if self.oob_score_constrained:
            self._set_oob_score_constrained(self.train_X,self.train_Y)
                
    def _set_oob_score_constrained(self, X, y):
        """Compute out-of-bag score"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        #n_classes_ = self.rfc.n_classes_
        n_samples = y.shape[0]

#        oob_decision_function = []
#        oob_score = 0.0
#        #predictions = []
#        for k in range(self.rfc.n_outputs_):
#            predictions.append(np.zeros((n_samples, n_classes_[k])))
            
        unsampled_indices=[]
        leafpreds=self.leaf_predictions_unc if self.__pm_weights is None else self.leaf_predictions
        numTrees=np.int(self.rfc.get_params()['n_estimators'])
        n=len(self.train_X)
        if len(X.shape)==1:
            X=X.reshape(1,-1) # ensure 2D array
        test_leafnodes_all=self.rfc.apply(X)
        # load unsampled indices
        for estimator in self.rfc.estimators_: #
            unsampled_indices.append(_generate_unsampled_indices(estimator.random_state, n_samples))
        # use kNN approach to predict RF class - only use trees where data point is not part of the training
        results=np.zeros(n_samples)
        for ksample in np.arange(n_samples):
            res=0.0
            test_leafnodes=test_leafnodes_all[ksample,:]
            for jtree in np.arange(0,np.int(numTrees)):
                if ksample in unsampled_indices[jtree]:
                    res=res+leafpreds[test_leafnodes[jtree],jtree]
            results[ksample]=np.double(n)*res
        classes= np.sign(results)
        classes[classes==0.]=-1.    
        oob_score = np.mean(y ==classes)
        self.oob_score_constrained_ = oob_score
#        for estimator in self.rfc.estimators_: #UTH
#            unsampled_indices = _generate_unsampled_indices(
#                estimator.random_state, n_samples)
#            p_estimator = estimator.predict_proba(X[unsampled_indices, :],
#                                                  check_input=False)
#
#            if self.n_outputs_ == 1:
#                p_estimator = [p_estimator]
#
#            for k in range(self.n_outputs_):
#                predictions[k][unsampled_indices, :] += p_estimator[k]
#
#        for k in range(self.n_outputs_):
#            if (predictions[k].sum(axis=1) == 0).any():
#                warn("Some inputs do not have OOB scores. "
#                     "This probably means too few trees were used "
#                     "to compute any reliable oob estimates.")
#
#            decision = (predictions[k] /
#                        predictions[k].sum(axis=1)[:, np.newaxis])
#            oob_decision_function.append(decision)
#            oob_score += np.mean(y[:, k] ==
#                                 np.argmax(predictions[k], axis=1), axis=0)
#
#        if self.n_outputs_ == 1:
#            self.oob_decision_function_ = oob_decision_function[0]
#        else:
#            self.oob_decision_function_ = oob_decision_function
#
#        self.oob_score_constrained_ = oob_score / self.n_outputs_
    def calc_forest_overlap(self):
        n=len(self.train_X)
        trees=self.rfc.estimators_
        unbiased_trees_for_each_pt=dict()
        # determine unbiased trees for each pt
        for i in np.arange(n):
            unbiased_trees=[]    
            for j in np.arange(self.num_trees):
                tree=trees[j]
                indexes_of_samples_not_used_for_this_tree=_generate_unsampled_indices(tree.random_state, self.num_samples)
                if i in indexes_of_samples_not_used_for_this_tree:
                    unbiased_trees.append(j)
            unbiased_trees_for_each_pt[str(i)]=unbiased_trees
        # determine overlap between points
        overlapping_unbiased_trees=np.zeros((n,n))
        for i_1 in np.arange(n):
            for i_2 in np.arange(n):
                overlapping_trees=[val for val in unbiased_trees_for_each_pt[str(i_1)] if val in unbiased_trees_for_each_pt[str(i_2)]]
                overlapping_unbiased_trees[i_1,i_2]=len(overlapping_trees)
        return overlapping_unbiased_trees
    
    def get_tree_samples(self,used_samples=True):
        trees=self.rfc.estimators_
        tree_samples=dict()
        itree=0
        for tree in trees:
            if used_samples:
                tree_samples[itree]=_generate_sample_indices(tree.random_state, self.num_samples)
            else:
                tree_samples[itree]=_generate_unsampled_indices(tree.random_state, self.num_samples)
            itree+=1
        return tree_samples
    
    def get_sample_trees(self,used_in_trees=True):
        n=len(self.train_X)
        numtrees=np.int(self.rfc.get_params()['n_estimators'])
        tree_samples=self.get_tree_samples(used_samples=used_in_trees)
        sample_trees=dict()
        for i in np.arange(n):
            trees=[]
            for j in np.arange(numtrees):
                if i in tree_samples[j]:
                    trees.append(j)
            sample_trees[i]=trees
        return sample_trees
    def est_performance_soft_penalty(self,constraints_lower,constraints_higher,lambda_constr_violation_weighting ):
        numConstr=len(constraints_lower)
        
        if numConstr==0:
            self.__pm_weights =None
        else:
            n=len(self.train_X)
            ys_pred=np.zeros(n)
            sample_unbiased_trees=self.get_sample_trees(used_in_trees=False)
            for i in np.arange(n):
                print('sample pt: ' + str(i))
                # solve constraints and predict point
                pmrf_copy=copy.copy(self)    
                pmrf_copy.oob_score_constrained=False
                pmrf_copy.fitConstrained_SoftPenalty(constraints_lower,constraints_higher,lambda_constr_violation_weighting,sample_unbiased_trees[i])
                ys_pred[i]=pmrf_copy.predictConstrained(self.train_X[i,:],useconstraints=True,use_trees=sample_unbiased_trees[i]) 
            acc=sum(ys_pred==self.train_Y)/len(self.train_Y)
            return acc
            
            
    def find_lin_indep_cols_mask(self,matrix,diff_threshold=1e-4):
        # Returns a linearly independent set of the columns of matrix. For linearly dependent columns the last dependent column 
        # will be included, but the other(s) excluded, so that the returned columns compose a full basis set for the column space.
        
        # identify all columns
        all_zero_vals=[np.all(matrix[:,i]==0) for i in range(matrix.shape[1])]
        # find independent cols
        indep_indices=np.zeros(matrix.shape[1],dtype='bool')
        for i in range(matrix.shape[1]):
            independent=True
            for j in range(i+1,matrix.shape[1]):
                if  not all_zero_vals[i] and not all_zero_vals[j]:
                    inner_product = np.inner(matrix[:,i],matrix[:,j])
                    norm_i = np.linalg.norm(matrix[:,i])
                    norm_j = np.linalg.norm(matrix[:,j])
                    if np.abs(inner_product - norm_j * norm_i) < diff_threshold:
                        independent=False
                        break
            indep_indices[i]=False  if all_zero_vals[i] else independent
        return indep_indices
        

           
    def fitConstrained_SoftPenalty(self,constraints_lower_arg,constraints_higher_arg,lambda_constr_violation_weighting,use_trees='all',constr_penalty_wgts='equal'):
        self.__leaf_predictions=None
        numtrees=np.int(self.rfc.get_params()['n_estimators'])
        if len(constraints_higher_arg)==0:
            self.__pm_weights =None
        else:
            
            # see if any constraints are in the same leaf nodes for all trees
            test_leafnodes_higher_all_test=self.rfc.apply(constraints_higher_arg)
            test_leafnodes_lower_all_test=self.rfc.apply(constraints_lower_arg)
            # check constraits for no change between higher /lower, and duplicates:
            good_constr_indexes=check_constraints(test_leafnodes_lower_all_test,test_leafnodes_higher_all_test)
            constraints_higher=constraints_higher_arg[good_constr_indexes,:].copy()
            constraints_lower=constraints_lower_arg[good_constr_indexes,:].copy()
            test_leafnodes_higher_all=test_leafnodes_higher_all_test[good_constr_indexes,:].copy()
            test_leafnodes_lower_all=test_leafnodes_lower_all_test[good_constr_indexes,:].copy()
            if constraints_higher_arg.shape[0]>constraints_higher.shape[0]:
                print('Duplicates and/or stationary constraints found. Orig: ' + str(constraints_higher_arg.shape[0])  + ' reduced: ' + str(constraints_higher.shape[0]))
            numConstr=len(constraints_lower)
            if numConstr==0:
                self.__pm_weights =None
            else: #numConstr>0 - ie we hve some constraints!
                # Build QP matrices
                #Minimize     1/2 x^T G x - a^T x
                #Subject to   C.T x >= b    
                n=len(self.train_X)
                train_Y_arr=np.double(np.ravel(self.train_Y))
                trees= np.arange(0,np.int(numtrees)) if use_trees=='all' else use_trees                        
                
                # calculate constraint penalty weights
                if constr_penalty_wgts=='equal':
                    s=np.ones([numConstr,1])
                elif constr_penalty_wgts=='inverse_diff':
                    pred_higher=self.predictConstrainedReal(constraints_higher,useconstraints=False) 
                    pred_lower=self.predictConstrainedReal(constraints_lower,useconstraints=False) 
                    diff=pred_lower-pred_higher
                    diff[diff<=0]=1.0
                    s=1.0/np.exp(diff)
                    s=s.reshape([len(s),1])
                      
                # Build intermediate matrix L
                L=np.zeros((n,numConstr))
                for kconstr in np.arange(numConstr):
                    test_leafnodes_higher=test_leafnodes_higher_all[kconstr,:]
                    test_leafnodes_lower=test_leafnodes_lower_all[kconstr,:]
                    for i in np.arange(0,len(self.train_X)):             
                        res=0.0
                        for jtree in trees:
                            thistree_trainptleafnode=self.training_leafnodes[i,jtree]
                            thistree_trainptleafnode_count=self.training_sample_counts[i,jtree]
                            # do higher constraint value with PLUS
                            thistree_testleafnode_higher=test_leafnodes_higher[jtree]
                            if thistree_testleafnode_higher==thistree_trainptleafnode:
                                res=res+train_Y_arr[i]*1.0*thistree_trainptleafnode_count/np.double(numtrees)/np.double(self.training_leaf_populations[jtree][thistree_testleafnode_higher])#removed: -1
                            # do lower constraint value with MINUS
                            thistree_testleafnode_lower=test_leafnodes_lower[jtree]
                            if thistree_testleafnode_lower==thistree_trainptleafnode:
                                res=res-train_Y_arr[i]*1.0*thistree_trainptleafnode_count/np.double(numtrees)/np.double(self.training_leaf_populations[jtree][thistree_testleafnode_lower])#removed: -1
                        L[i,kconstr]=res
                # check L for duplicate columns which lead to no positive determinate G. This is due to duplicate constraints (in terms of leaf nodes)
                if L.shape[1]>L.shape[0]:
                    print('WARNING: more constraints than data points. Constraints will be culled to less than the number of datapoints using reduced echeleon form for matrix L')
                rank =np.linalg.matrix_rank(L)
                if rank<L.shape[1]:
                    #constr_mask=self.find_lin_indep_cols_mask(L)
                    cont=True
                    thresh=1e-7
                    while cont:
                        zero= lambda arg: arg<thresh
                        [R,constr_mask]=sympy.Matrix(L).rref(zero)
                        if len(constr_mask)<=rank:
                            cont=False
                        else:# have not reduced to rank yet
                            thresh=thresh*10
                            print('Increasing zero threshold on rref...')
                else: # full rank
                    #constr_mask=np.ones(L.shape[1],dtype='bool') # ie all constraints
                    constr_mask=range(L.shape[1])
                # update effective constraints if any linearly dependent cols found
                L_orig=L.copy()
                orig_constr_num=L.shape[1]
                L=L[:,constr_mask]
                if L.shape[1]<orig_constr_num:
                    print('Matrix L contained linearly dependent cols (corresponding to constraints): orig: ' + str(orig_constr_num)  + ' reduced cols: ' + str(L.shape[1]))
                if L.shape[1]==0:
                    print('Matrix L is empty!')
                constraints_lower_effective=constraints_lower[constr_mask,:]
                constraints_higher_effective=constraints_higher[constr_mask,:]
                s_effective=s[constr_mask,:]
                numConstr_effective=constraints_lower_effective.shape[0]
                test_leafnodes_higher_all_effective=test_leafnodes_higher_all[constr_mask,:]
                test_leafnodes_lower_all_effective=test_leafnodes_lower_all[constr_mask,:]
                #print(L)
                # Construct solution matrices
                ones=np.ones((n,n),dtype='float')
                G=-0.5*1.0/n/numtrees**2*np.dot(np.dot(L.T,ones),L)+0.5*np.dot(L.T,L)
                a=-1.0*np.dot(np.ones((1,n)),L).T
                C=np.vstack((np.eye(numConstr_effective),-1*np.eye(numConstr_effective))).T
                b=np.vstack((np.zeros((numConstr_effective,1)),-s_effective*lambda_constr_violation_weighting))
    #            # check matrix rank:
                rank =np.linalg.matrix_rank(G)
                if rank<G.shape[0]:
                    print('G is not full rank: ' + str(rank) + ' vs ' + str(G.shape[0]))
    #            unique_a = np.vstack({tuple(row) for row in G})
    #            print('numconstr:' + str(G.shape[0]) + ' unique: ' + str(unique_a.shape[0]))
                # check eigvals - if there are any negative, it is not positive definite
                evals, evecs = np.linalg.eig(G)
                idx = evals.argsort()   
                evals = evals[idx] 
                mineigval=evals[0]
                if mineigval<0: # add Tikhonov regularisation to ensure positive definite
                    print('Negative mineigval found: ' + str(mineigval))
    #                print(G)
    #                G=G-np.eye(G.shape[0])*2.0*mineigval
    #                print(G)
    #                evals, evecs = np.linalg.eig(G)
    #                idx = evals.argsort()   
    #                evals = evals[idx] 
    #                mineigval=evals[0]
    #                if mineigval<0:
    #                    print('Min eig val still negative!!: ' + str(mineigval))
                # Solve QP for new sample weights
                p=solve_qp(G,a.ravel(),C,b.ravel(),0,False)
                alphas=p[0]
                Cis=np.zeros((n,1))
                for kconstr in np.arange(numConstr_effective):
                    test_leafnodes_higher=test_leafnodes_higher_all_effective[kconstr,:]
                    test_leafnodes_lower=test_leafnodes_lower_all_effective[kconstr,:]
                    for i in np.arange(0,len(self.train_X)):             
                        for jtree in trees:
                            thistree_trainptleafnode=self.training_leafnodes[i,jtree]
                            thistree_trainptleafnode_count=self.training_sample_counts[i,jtree]
                            # do higher constraint value with PLUS
                            thistree_testleafnode_higher=test_leafnodes_higher[jtree]
                            if thistree_testleafnode_higher==thistree_trainptleafnode:
                                Cis[i]=Cis[i]+alphas[kconstr]*train_Y_arr[i]*1.0*thistree_trainptleafnode_count/np.double(numtrees)/np.double(self.training_leaf_populations[jtree][thistree_testleafnode_higher]) #removed: -1
                            # do lower constraint value with MINUS
                            thistree_testleafnode_lower=test_leafnodes_lower[jtree]
                            if thistree_testleafnode_lower==thistree_trainptleafnode:
                                Cis[i]=Cis[i]-alphas[kconstr]*train_Y_arr[i]*1.0*thistree_trainptleafnode_count/np.double(numtrees)/np.double(self.training_leaf_populations[jtree][thistree_testleafnode_lower]) #removed: -1
                        
                gamma=1/n/numtrees*np.sum(Cis)     
                ps=1.0/n-gamma/2.0+1/2.0/numtrees*Cis
                self.pm_weights= ps # extract coefficients
                #print(p[0][p[0]<0.99*(1.0/np.double(n))])
                #return 0
        if self.oob_score_constrained:
            self._set_oob_score_constrained(self.train_X,self.train_Y)
            
            
from pmforest import PMRandomForestClassifier
class model_pmrf_c:
    def __init__(self,obj_PMRandomForestClassifier):
        self.clf=obj_PMRandomForestClassifier
    def predict(self,Xs,opt_use_constrained=True):
       	return self.clf.predictConstrained(Xs,opt_use_constrained)

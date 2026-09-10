from aegis_rad.evaluation.structured import ufr,supported_recall,grounding_f1,critical_omission

def f(name,region=None,state='present'):return {'name':name,'region':region,'state':state}

def test_structured_metrics():
    ref=[f('Cardiomegaly','Cardiac silhouette'),f('Pleural Effusion','Right costophrenic')]
    pred=[f('Cardiomegaly','Cardiac silhouette'),f('Pneumothorax','Right apical')]
    assert abs(ufr(pred,ref)-.5)<1e-8
    assert abs(supported_recall(pred,ref)-.5)<1e-8
    assert abs(grounding_f1(pred,ref)-.5)<1e-8
    assert critical_omission(pred,ref,['Pleural Effusion'])==1.0

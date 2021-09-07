class HyperParameters:
    def __init__(self,
                 hyperparam_dict : dict):
        
        self.__dict__ = hyperparam_dict
        for key, val in self.dict_hp.items():
            setattr(self, key, val)
    
    def __str__(self):
        
        template = "name [ type ] : value \n"
        template += "="*20 + "\n"
        for  key, val in self.__dict__.items():
            template += f"{key} [{type(val)}] : {val} \n"
        
        return template
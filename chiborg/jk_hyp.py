class jk_hyp():

    def __init__(mode="diagonal"):
        valid_modes = ["diagonal", "partition", "manual"]
        if mode not in :
            raise ValueError(f"mode keyword must be one of {valid_modes}")
        

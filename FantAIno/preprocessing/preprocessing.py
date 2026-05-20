class PreprocessingPipeline:
    """
    Preprocessing pipeline for FantAIno.
    """
    
    def __init__(self, steps):
        self.steps = steps

    def __call__(self, df):
        for step in self.steps:
            df = step(df)
        return df
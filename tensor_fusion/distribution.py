import torch.distributions as dist

class LogUniform(dist.TransformedDistribution):
    '''
    Exp(Uniform(log(lower_bound), log(upper_bound)))
    '''
    def __init__(self, lower_bound, upper_bound):
        super(LogUniform, self).__init__(dist.Uniform(lower_bound.log(), upper_bound.log()),
                                         dist.ExpTransform())
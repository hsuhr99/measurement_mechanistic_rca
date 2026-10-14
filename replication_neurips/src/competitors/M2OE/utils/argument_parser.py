from argparse import ArgumentParser

class DefaultConfig(object):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        """
        Defines all the arguments for running an experiment.
        :param parser: instance of an ArgumentParser.
        :return: the parser with added arguments
        """

        # define directories for datasets and logging
        parser.add_argument(
            '--logdir', type=str, default='.',
            help='Directory where log data is to be stored. '
        )

        parser.add_argument(
            '--readme', type=str, default='',
            help='Some notes to be stored in the automatically created config.txt configuration file.'
        )

        # test parameters
        parser.add_argument('-eb', '--exp-batch', type=int, default=16)
        parser.add_argument('-ee', '--exp-epochs', type=int, default=30)
        parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
        parser.add_argument('-no', '--n-other', type=int, default=1,
                            help='number of samples of the other class, must be less or equals to the number'
                                 'of samples in that class')
        parser.add_argument('-t', '--threshold', type=float, default=0.5,
                            help='Threshold for which a dimension is anomalous')

        parser.add_argument(
            '--loss-weights', type=float, nargs='*', default=[1., 1., .0, .5],
            help='Weights of the losses used for  '
        )
        parser.add_argument(
            '--gpu-to-use', type=int, default=0, help='Number of the GPU to use'
        )

        return parser


class MultipleRuns(DefaultConfig):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        parser = super().__call__(parser)
        parser.add_argument('--n-runs', type=int, default=5)

        return parser


class RealRuns(MultipleRuns):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        parser = super().__call__(parser)
        parser.add_argument('--dataset', type=str)

        return parser   
    
    
class DefaultGenerated(DefaultConfig):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        parser = super().__call__(parser)
        parser.add_argument('-nd', '--dim-number', type=int, default=10)
        parser.add_argument('-nad', '--a-dim-number', type=int, default=1)
        parser.add_argument('-nns', '--n-samples-num', type=int, default=100)
        parser.add_argument('-nas', '--a-samples-num', type=int, default=1)
        parser.add_argument('-nm', '--n-mean', type=float, default=3)
        parser.add_argument('-nstd', '--n-std', type=float, default=0.3)
        parser.add_argument('-dstd', '--dist-std', type=int, default=9,
                            help='Number of standard deviations in which the mean of the anomaly '
                                 'will be placed, this number must be grater than 8 otherwise'
                                 'there is a possible overlap between'
                                 'Used only in correlation test')
        #parser.add_argument('-dstd2', '--dist-std-2', type=int,
        #                    help='Number of standard deviations in which the mean of the anomaly '
        #                         'will be placed, this number must be grater than 8 otherwise'
        #                         'there is a possible overlap between'
        #                         'Used only in correlation test')
        parser.add_argument('--anom-dims', type=int,
                            help='Number of dimentions along with the sample is anomalous', default=1)


        return parser


class DefaultGeneratedMultiple(DefaultConfig):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--n-runs', type=int, default=5)
        parser = super().__call__(parser)
        parser.set_defaults(
            a_mean=1, a_std=0.06, a_dim_number=1, anom_dims=1)
        return parser

class DefaultGeneratedCorr(DefaultConfig):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        parser = super().__call__(parser)

        parser.add_argument(
                 '--centers', type=float, nargs='*', default=[[1.2, 6.7], [5.8, 3.6]],
                 help='Weights of the losses used for  '
            )

        parser.set_defaults(
            c_mean=6, c_std=0.9, corr_anom='gen',
            corr_deg=0.7, dist_std_2=5
        )
        return parser


class DefaultReal(DefaultConfig):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        parser = super().__call__(parser)
        parser.add_argument('--dataset', type=str,
                            help='Dataset on which perform test', default='wbc')
        return parser

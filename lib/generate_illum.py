from numpy import ones, exp, power, linspace
from colour import SpectralDistribution


class SyntheticIlluminant(SpectralDistribution):
    def __init__(self, domain=list(range(300, 801)), base_power=0, name='Synthetic Illuminant'):
        super().__init__(domain=domain, data=ones((len(domain)))*base_power, name=name)

    def gauss(self, x, mu, sig):
        return exp(-power(x - mu, 2.) / (2 * power(sig, 2.)))

    def add_gauss(self, sigma, center, intensity=1):
        x = self.domain
        signal = SpectralDistribution(
            domain=x, data=intensity*self.gauss(x, center, sigma))
        addition = self + signal

        return addition

    def subtract_gauss(self, sigma, center, intensity=1):
        x = self.domain
        signal = SpectralDistribution(
            domain=x, data=intensity*self.gauss(x, center, sigma))
        subtr = self - signal

        return subtr

from matplotlib import pyplot as plt
from numpy import ones, mean, abs, zeros
from cv2 import resize, circle
from math import floor
from random import random
from colour import SpectralDistribution, SpectralShape


class Spectral:
    im = None
    white = None
    dark = None
    wavelengths = None
    dim_orig = None
    dim_cropped = None
    dim_scaled = None
    crop = None
    # should inputs (pixel locations) be scaled along with image; useful for quick iterating, before using full image for final renders
    scale_ratio = 1
    scale_inputs = None
    trim = None

    def __init__(self, im, white, dark, scale=None, scale_inputs=True, crop=None, trim=None):
        self.wavelengths = im[1]

        if scale is not None:
            self.scale_ratio = scale
        self.scale_inputs = scale_inputs
        self.dim_orig = im[0].shape
        # avg and resize white/dark refs
        i = im[0]
        w = white[0]
        d = dark[0]

        if trim is not None:
            self.trim = trim
            idx_start = (abs(self.wavelengths - trim[0])).argmin()
            if self.wavelengths[idx_start] > trim[0] and idx_start > 0:
                idx_start = idx_start - 1

            idx_end = (abs(self.wavelengths - trim[1])).argmin()
            if self.wavelengths[idx_end] < trim[1] and idx_end < len(self.wavelengths - 1):
                idx_end = idx_end + 1

            i = i[:, :, idx_start:idx_end+1]
            w = w[:, :, idx_start:idx_end+1]
            d = d[:, :, idx_start:idx_end+1]
            self.wavelengths = self.wavelengths[idx_start:idx_end+1]

        if crop is not None:
            self.crop = crop
            i = i[crop[0]:crop[1], crop[2]:crop[3], :]
            self.dim_cropped = i.shape

        if scale is not None:
            i = self.scale(i, scale)
            w = self.scale(w, scale)
            d = self.scale(d, scale)
            self.dim_scaled = i.shape

        w, d = self.prepare_light_dark(i, w, d)
        self.white = w
        self.dark = d

        # remove dark noise
        i = (i - d)/(w - d)
        self.im = i

    def scale(self, im, ratio):
        x = floor(im.shape[1]*ratio)
        y = floor(im.shape[0]*ratio)
        return resize(im, (x, y))

    def take_avg(self, im):
        return mean(im, axis=(0, 1))

    def prepare_light_dark(self, im, white, dark):
        w = self.take_avg(white)
        d = self.take_avg(dark)
        w = ones(im.shape)*w
        d = ones(im.shape)*d
        return (w, d)

    def nearest_band_index(self, wavelength):
        idx = (abs(self.wavelengths - wavelength)).argmin()
        return idx

    def rand_composite(self):
        num = len(self.wavelengths)
        return (floor(random()*num), floor(random()*num), floor(random()*num))

    def resize_inputs(self, location, scalars):
        loc_new = (location[0] - self.crop[2], location[1] -
                   self.crop[0]) if self.crop is not None else location
        loc_new = (floor(loc_new[0]*self.scale_ratio),
                   floor(loc_new[1]*self.scale_ratio))

        scalars_new = []
        for s in scalars:
            scalars_new.append(floor(s*self.scale_ratio))

        return (loc_new, scalars_new)

    def get_spectra(self, location: tuple, sample_radius):
        if self.scale_inputs is not None:
            location, sample_radius = self.resize_inputs(
                location, [sample_radius])
            sample_radius = sample_radius[0]

        mask = zeros(self.im.shape[:2])
        mask = circle(mask, location, sample_radius, color=255, thickness=-1)
        spectral_patch = self.im[mask == 255]
        return mean(spectral_patch, axis=0)

    def show_bands(self, bands, show, save=False, composite=False, use_band_index=True):
        if use_band_index == False:
            bands = tuple([self.nearest_band_index(b) for b in bands])

        wavelengths = [self.wavelengths[b] for b in bands]

        if composite == True and len(bands) == 3:
            print(f'Showing composite of bands {wavelengths}')
            pic = zeros((self.im.shape[0], self.im.shape[1], 3))
            pic[:, :, 0] = self.im[:, :, bands[0]]
            pic[:, :, 1] = self.im[:, :, bands[1]]
            pic[:, :, 2] = self.im[:, :, bands[2]]
            show(
                pic, filename=f'spec-recon_{bands[0]}-{bands[1]}-{bands[2]}.png' if save is True else None)
        else:
            print('Showing bands', end='', flush=True)
            for idx, b in enumerate(bands):
                print(' ', b, end='', flush=True)
                pic = self.im[:, :, b]
                show(
                    pic, filename=f'spec-recon_{b}.png' if save is True else None)
            print()

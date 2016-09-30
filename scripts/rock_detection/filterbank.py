# -*- coding: utf-8 -*-

'''
    filterbankを返すスクリプト
'''
import numpy as np

radius = 5

def main(name, radius):

    if name == 'MR':   bank = mr.main(radius)
    elif name == 'LM': bank = lm.main(radius)         
    else: print 'There is not {} filter bank'.format(name)

    return bank

class MaximumResponsesSet():

    def main(self, radius):
        """ Generates filters for RFS filterbank.
        Parameters
        ----------
        radius : int, default 28
            radius of all filters. Size will be 2 * radius + 1
        sigmas : list of floats, default [1, 2, 4]
            define scales on which the filters will be computed
        n_orientations : int
            number of fractions the half-angle will be divided in
        Returns
        -------
        edge : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
            Contains edge filters on different scales and orientations
        bar : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
            Contains bar filters on different scales and orientations
        rot : ndarray (2, 2*radius+1, 2*radius+1)
            contains two rotation invariant filters, Gaussian and Laplacian of
            Gaussian
        """

        sigmas = [1, 2, 4]
        n_orientations = 6

      
        def make_gaussian_filter(x, sigma, order=0):
            if order > 2:
                raise ValueError("Only orders up to 2 are supported")

            # compute unnormalized Gaussian response
            response = np.exp(-x ** 2 / (2. * sigma ** 2))
            if order == 1:
                response = -response * x
            elif order == 2:
                response = response * (x ** 2 - sigma ** 2)

            # normalize
            response /= np.abs(response).sum()
            return response

        def makefilter(scale, phasey, pts, sup):
            gx = make_gaussian_filter(pts[0, :], sigma=3 * scale)
            gy = make_gaussian_filter(pts[1, :], sigma=scale, order=phasey)
            f = (gx * gy).reshape(sup, sup)
            
            # normalize
            f /= np.abs(f).sum() # L1 norm

            return f

        support = 2 * radius + 1
        x, y = np.mgrid[-radius:radius + 1, radius:-radius - 1:-1]
        orgpts = np.vstack([x.ravel(), y.ravel()])

        rot, edge, bar = [], [], [] # 空の用意
        for sigma in sigmas:
            for orient in xrange(n_orientations):
                # Not 2pi as filters have symmetry
                angle = np.pi * orient / n_orientations
                c, s = np.cos(angle), np.sin(angle)
                rotpts = np.dot(np.array([[c, -s], [s, c]]), orgpts)
                edge.append(makefilter(sigma, 1, rotpts, support))
                bar.append(makefilter(sigma, 2, rotpts, support))
        length = np.sqrt(x ** 2 + y ** 2)
        rot.append(make_gaussian_filter(length, sigma=10)) # gaussian filter
        rot.append(make_gaussian_filter(length, sigma=10, order=2)) # log filter

        bank = []
        bank.append(edge)
        bank.append(bar)
        bank.append(rot)

        return bank

class LeungMalikSet():

    def main(self, size):
        params = [[2,1],[4,1],[4,2],[6,1],[6,2],[6,3],[8,1],[8,2],[8,3],[10,1],[10,2],[10,3],[10,4]]
        kernels = []
        
        def  makefilter(sup, sigma, tau):
            hsup = (sup-1)/2
            x,y = np.meshgrid(range(-hsup,sup-hsup),range(-hsup,sup-hsup))
            r = np.sqrt(x**2 + y**2)
            f = np.cos(r*(np.pi*tau/sigma))*np.exp(-(r*r)/(2*sigma*sigma))
            f = f - f.mean()          # Pre-processing: zero mean
            f = f / abs(f).sum()      # Pre-processing: L_{1} normalise
            return f
            
        for param in params:
            kernel = makefilter(size, param[0], param[1])
            kernels.append(kernel)

        return np.array(kernels)

mr = MaximumResponsesSet()
lm = LeungMalikSet()

if __name__ == '__main__':
    
    main('LM', radius=5)

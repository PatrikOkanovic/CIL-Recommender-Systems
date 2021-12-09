from lib.models import autoencoder, autorec, fm, kernel_net, ncf, nmf, svd

models = {'svd': svd,
          'nmf': nmf,
          'kernel_net': kernel_net,
          'autoencoder': autoencoder,
          'autorec': autorec,
          'fm': fm,
          'ncf': ncf
          }

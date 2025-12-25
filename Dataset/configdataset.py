import os
import pickle

DATASETS = [ 'roxford5k', 'rparis6k','revisitop1m']


def RoxfordAndRparis(dataset, dir_main):
    dataset = dataset.lower()

    if dataset not in DATASETS:
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    # loading imlist, qimlist, and gnd, in cfg as a dict
    if dataset == 'roxford5k' or dataset == 'rparis6k':
        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'
    if dataset == 'revisitop1m':
        cfg = {}
        cfg['imlist_fname'] = os.path.join(dir_main, dataset, '{}.txt'.format(dataset))
        cfg['imlist'] = read_imlist(cfg['imlist_fname'])
        cfg['qimlist'] = []
        cfg['ext'] = ''
        cfg['qext'] = ''
    cfg['dir_data'] = os.path.join(dir_main, dataset)
    cfg['dir_images'] =cfg['dir_data']
    #cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = []
    cfg['im_fname1'] = []
    if dataset == 'revisitop1m':
        for name in cfg['imlist']:
            cfg['im_fname1'].append(os.path.join(cfg['dir_images'], name))

    else:
        for name in cfg['imlist']:
            cfg['im_fname'].append(os.path.join(cfg['dir_images'], name + '.jpg'))


    cfg['qim_fname'] = []
    for name in cfg['qimlist']:
        cfg['qim_fname'].append(os.path.join(cfg['dir_images'], name + '.jpg'))

    cfg['dataset'] = dataset

    return cfg


def read_imlist(imlist_fn):
    with open(imlist_fn, "r") as file:
        imlist = file.read().splitlines()
    return imlist
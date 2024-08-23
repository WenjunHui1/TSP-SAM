class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'MoCA':
            return '../../dataset/MoCA-Mask/' 
        elif dataset == 'CAD2016':
            return '../../dataset/CAD2016/CamouflagedAnimalDataset/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

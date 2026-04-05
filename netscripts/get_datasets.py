from datasets.datasetopts import DatasetArgs
from datasets.holoaders import EpicHODataset as HODataset, FeaturesHOLoader, get_dataloaders


def get_dataset(args, base_path="./"):
    if args.evaluate:
        mode = "validation"
    else:
        mode = 'train'

    datasetargs = DatasetArgs(ek_version=args.ek_version, mode=mode,
                              use_label_only=True, base_path=base_path,
                              batch_size=args.batch_size, num_workers=args.workers)
    if getattr(args, 'use_gaze', False):
        datasetargs.use_gaze = True
        datasetargs.gaze_heatmap_size = getattr(args, 'gaze_heatmap_size', 32)
        datasetargs.gaze_sigma = getattr(args, 'gaze_sigma', 2.0)
    dls = get_dataloaders(datasetargs, HODataset, featuresloader=FeaturesHOLoader)
    return mode, dls


from dataset.t2i import build_t2i, build_t2i_code, build_t2i_image
from dataset.t2v import build_t2v
from dataset.t2iv import build_t2iv


def build_dataset(args, **kwargs):
    if args.dataset == 't2v':
        return build_t2v(args, **kwargs)
    if args.dataset == 't2iv':
        return build_t2iv(args, **kwargs)
    raise ValueError(f'dataset {args.dataset} is not supported')
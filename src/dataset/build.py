from src.dataset.imagenet import build_imagenet, build_imagenet_code
from src.dataset.coco import build_coco
from src.dataset.openimage import build_openimage
from src.dataset.pexels import build_pexels
from src.dataset.t2i import build_t2i, build_t2i_code, build_t2i_image


def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    if args.dataset == 'coco':
        return build_coco(args, **kwargs)
    if args.dataset == 'openimage':
        return build_openimage(args, **kwargs)
    if args.dataset == 'pexels':
        return build_pexels(args, **kwargs)
    if args.dataset == 't2i_image':
        return build_t2i_image(args, **kwargs)
    if args.dataset == 't2i':
        return build_t2i(args, **kwargs)
    if args.dataset == 't2i_code':
        return build_t2i_code(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')
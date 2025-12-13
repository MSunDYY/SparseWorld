# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from functools import partial
from pathlib import Path
from mmcv import ConfigDict
import numpy as np
import torch
import pdb
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmdet3d.models import build_model
from mmcv.cnn import MODELS
from mmseg.datasets import build_dataloader as build_mmseg_dataloader
try:
    from mmengine.analysis import get_model_complexity_info, parameter_count
    # from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')
from mmdet3d.datasets import build_dataset

def _format_size(x: int, sig_figs: int = 3, hide_zero: bool = False) -> str:
    """Formats an integer for printing in a table or model representation.

    Expresses the number in terms of 'kilo', 'mega', etc., using
    'K', 'M', etc. as a suffix.

    Args:
        x (int): The integer to format.
        sig_figs (int): The number of significant figures to keep.
            Defaults to 3.
        hide_zero (bool): If True, x=0 is replaced with an empty string
            instead of '0'. Defaults to False.

    Returns:
        str: The formatted string.
    """
    if hide_zero and x == 0:
        return ''

    def fmt(x: float) -> str:
        # use fixed point to avoid scientific notation
        return f'{{:.{sig_figs}f}}'.format(x).rstrip('0').rstrip('.')

    # if abs(x) > 1e14:
    #     return fmt(x / 1e15) + 'P'
    # if abs(x) > 1e11:
    #     return fmt(x / 1e12) + 'T'
    # if abs(x) > 1e8:
    #     return fmt(x / 1e9) + 'G'
    if abs(x) > 1e5:
        return fmt(x / 1e6) + 'M'
    if abs(x) > 1e2:
        return fmt(x / 1e3) + 'K'
    return str(x)

def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='num images of calculate model flops')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference(args, logger):
    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    # if 'train_dataloader' not in cfg.data:
    #     cfg.data['train_dataloader'] = ConfigDict()
    # if 'val_dataloader' not in cfg.data:
    #     cfg.data['val_dataloader'] = ConfigDict()
    # if 'test_dataloader' not in cfg.data:
    #     cfg.data['test_dataloader'] = ConfigDict()
    # cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # TODO: The following usage is temporary and not safe
    # use hard code to convert mmSyncBN to SyncBN. This is a known
    # bug in mmengine, mmSyncBN requires a distributed environment，
    # this question involves models like configs/strong_baselines
    if hasattr(cfg, 'head_norm_cfg'):
        cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)

    result = {}
    # avg_flops = []
    # data_loader = Runner.build_dataloader(cfg.val_dataloader)
    # val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    # data_loader = build_mmseg_dataloader(
    #     val_dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     shuffle=False)
    # model = MODELS.build(cfg.model)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    # _forward = model.forward

    # for idx, data_batch in enumerate(data_loader):
    #     if idx == args.num_images:
    #         break
    #     # data = model.data_preprocessor(data_batch)
    #     data = data_batch
    #     # pdb.set_trace()
    #     # print(data.keys())
    #     # pdb()
    #     # result['ori_shape'] = data['data_samples'][0].ori_shape
    #     # result['pad_shape'] = data['data_samples'][0].pad_shape
    #     # if hasattr(data['data_samples'][0], 'batch_input_shape'):
    #     #     result['pad_shape'] = data['data_samples'][0].batch_input_shape

    #     model.forward = partial(_forward, data_samples=data)
    #     outputs = get_model_complexity_info(
    #         model,
    #         None,
    #         inputs=data,
    #         show_table=False,
    #         show_arch=False)
    #     avg_flops.append(outputs['flops'])
    #     params = outputs['params']
    #     result['compute_type'] = 'dataloader: load a picture from the dataset'
    # del data_loader

    # mean_flops = _format_size(int(np.average(avg_flops)))
    # params = _format_size(params)
    # result['flops'] = mean_flops
    params = parameter_count(model)['']
    params = _format_size(params)

    result['params'] = params

    return result


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)
    split_line = '=' * 30
    # ori_shape = result['ori_shape']
    # pad_shape = result['pad_shape']
    # flops = result['flops']
    params = result['params']
    # compute_type = result['compute_type']

    # if pad_shape != ori_shape:
    #     print(f'{split_line}\nUse size divisor set input shape '
    #           f'from {ori_shape} to {pad_shape}')
    print(f'Params: {params}\n{split_line}')
    # print('!!!Please be cautious if you use the results in papers. '
    #       'You may need to check if all ops are supported and verify '
    #       'that the flops computation is correct.')


if __name__ == '__main__':
    main()

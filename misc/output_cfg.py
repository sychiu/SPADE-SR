import os
import yaml
import copy

with open('./train.yaml', "r") as stream:
    base_cfg = yaml.load(stream)

gp_weights = [5.0,0.5]

count = 0
for rec_mode in rec_modes:
    for d_ch in d_chs:
        for d_use_fake in d_use_fakes:
            for rec_weight in rec_weights:
                for g_z_dim in g_z_dims:
                    cfg = copy.deepcopy(base_cfg)
                    filename = 'test_%i' % count
                    cfg['filename'] = filename
                    cfg['RECONSTRUCTION_LOSS'] = rec_mode
                    cfg['D_CONV_CH'] = d_ch
                    cfg['D_USE_FAKE_RL'] = d_use_fake
                    cfg['G_REC_WEIGHT'] = rec_weight
                    cfg['D_REC_WEIGHT'] = rec_weight
                    cfg['G_Z_DIM'] = g_z_dim
                    count+=1
                    with open('./cfgs/%s.yaml'%filename, 'w') as file:
                        yaml.dump(cfg, file)

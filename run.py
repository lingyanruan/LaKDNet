'''
This source code is licensed under the license found in the LICENSE file.
This is the implementation of the "LaKDNet: Revisiting Image Deblurring with an Efficient ConvNet". 
Project GitHub repository: https://github.com/lingyanruan/LaKDNet
Email: lyruanruan@gmail.com
Copyright (c) 2024-present, Lingyan Ruan
'''

import os
from datetime import datetime
import torch
import torchvision.utils as vutils
from util.util import *
from pathlib import Path
import lpips
from glob import glob
from natsort import natsorted
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from models.LaKDNet import *
import argparse

parser = argparse.ArgumentParser(description='Defocus or Motion Testing')
parser.add_argument('--type', type=str, help='Defocus | Motion')
args = parser.parse_args()

import yaml

x = './options/Test_configs.yml'
type = args.type   #'Defocus' # 'Motion'

with open(x, 'r') as file:
    config = yaml.safe_load(file)[type]
    
test_status = config['test_status']
eval_data =config['eval_data']
net_configs = config['net_configs']
#### metrics #################################
compute_lpips = lpips.LPIPS(net='alex').cuda()


def test(input_c_file_path_list,gt_file_path_list,input_r_file_path_list=None,input_l_file_path_list=None,net_config=None,net_weight=None,result_dir=None,net_dual=None):

    PSNR_total,SSIM_total,LPIPS_total = 0,0,0
    PSNR_score, SSIM_score, LPIPS_score = 0,0,0
    
    #### make directory ################################
    Path(os.path.join(results_dir, 'input' )).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_dir, 'output')).mkdir(parents=True, exist_ok=True)

    for i, filename in enumerate(input_c_file_path_list):
        # Read Image

        filename = os.path.split(filename)[-1]
        if net_dual: # indicate dual mode

            C = read_image(input_c_file_path_list[i], 65535.0)
            C = torch.FloatTensor(C.transpose(0, 3, 1, 2).copy()).cuda()
            C,h,w = crop_image(C,8,True)
            filename = os.path.split(filename)[-1]



            R,L = read_image(input_r_file_path_list[i], 65535.0), read_image(input_l_file_path_list[i], 65535.0)
            R,L = crop_image(torch.FloatTensor(R.transpose(0, 3, 1, 2).copy()).cuda()), crop_image(torch.FloatTensor(L.transpose(0, 3, 1, 2).copy()).cuda())

            GT = read_image(gt_file_path_list[i], 65535.0)  # here to [0,1]
            GT = crop_image(torch.FloatTensor(GT.transpose(0, 3, 1, 2).copy()).cuda())

        else:

            C = read_image(input_c_file_path_list[i], 255.0)
            C = torch.FloatTensor(C.transpose(0, 3, 1, 2).copy()).cuda()
            C,h,w = crop_image(C,8,True)
            GT = read_image(gt_file_path_list[i], 255.0)  # here to [0,1]
            GT = crop_image(torch.FloatTensor(GT.transpose(0, 3, 1, 2).copy()).cuda())

        ##test resut
        with torch.no_grad():

            network = LaKDNet(**net_config).cuda()
            network.load_state_dict(torch.load(net_weight))
            
            if not net_dual:
                output = network(C)

            else:
                input =  torch.cat([L, R, C], 1).cuda()
                output = network(input)

        output =output[:,:,:h,:w]
        output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0) # to [0,1] for psnr and ssim evaluation

        save_file_path_deblur_input = os.path.join(result_dir, 'input',  '{}'.format(filename))
        save_file_path_deblur = os.path.join(result_dir, 'output', '{}'.format(filename))
        # vutils.save_image(C, '{}'.format(save_file_path_deblur_input), nrow=1, padding = 0, normalize = False)
        vutils.save_image(output, '{}'.format(save_file_path_deblur), nrow=1, padding = 0, normalize = False)
        # restored = np.uint16((restored*65535).round())

        if gt_file_path_list is not None and type=='Defocus':
            GT_cpu = GT.cpu().numpy()[0].transpose(1, 2, 0) 
            PSNR_score = compute_psnr(output_cpu, GT_cpu,data_range=1.0)
            
            SSIM_score = compute_ssim(output_cpu, GT_cpu,data_range=1.0,channel_axis=-1)
            LPIPS_score = compute_lpips(output*2-1, GT * 2. - 1.).item()

            # Log
            print('[EVAL][{:02}/{}] {} PSNR: {:.5f}, SSIM: {:.5f}, LPIPS: {:.5f}'.format( i + 1, total_files, filename, PSNR_score, SSIM_score,  LPIPS_score))
            with open(os.path.join(result_dir, 'score.txt'), 'w' if i == 0 else 'a') as file:
                file.write('[EVAL][{:02}/{}] {} PSNR: {:.5f}, SSIM: {:.5f}, LPIPS: {:.5f} \n'.format( i + 1, total_files, filename, PSNR_score, SSIM_score,  LPIPS_score))
                file.close()

            PSNR_total += PSNR_score
            SSIM_total += SSIM_score
            LPIPS_total += LPIPS_score

            ###=============================== network parameters info =======================================#######
            PSNR_mean,SSIM_mean,LPIPS_mean = PSNR_total / total_files,SSIM_total / total_files, LPIPS_total/total_files
        else:
            PSNR_mean,SSIM_mean,LPIPS_mean =0,0,0
            
    with open(os.path.join(result_dir, 'score.txt'), 'w' if i == 0 else 'a') as file:
        file.write('[EVAL MEAN][{}] PSNR: {:.5f}, SSIM: {:.5f}, LPIPS: {:.5f} \n'.format( total_files, PSNR_mean, SSIM_mean,  LPIPS_mean))
        file.close()

# walk through dataset
for ind, test_element in enumerate(test_status):
    net_config = config[net_configs[ind]]
    net_weight = config['weight'][test_element]

    if 'dual' in test_element:
        net_mode = True
    else:
        net_mode = False

    print('------------- net config ----------',test_element,net_mode)

    folder_time = datetime.now().strftime('%Y-%m-%d_%H%M')
    eval_subset = eval_data[ind]
    
    if isinstance(eval_data[ind], list) or net_mode:
        for sub_element in eval_data[ind]:
            
            if sub_element == 'DPDD':
                input_c_file_path_list = natsorted(glob(os.path.join(config[sub_element]['dataroot_lq'], 'test_c','source', '*.png')))
                input_r_file_path_list = natsorted(glob(os.path.join(config[sub_element]['dataroot_lq'], 'test_r', 'source', '*.png')))
                input_l_file_path_list = natsorted(glob(os.path.join(config[sub_element]['dataroot_lq'], 'test_l', 'source','*.png')))
                gt_file_path_list = natsorted(glob(os.path.join(config[sub_element]['dataroot_gt'], 'test_c', 'target', '*.png')))
                
            else:
                input_c_file_path_list = natsorted(glob(os.path.join(config[sub_element]['dataroot_lq'], 'input', '*.png')))
                gt_file_path_list = natsorted(glob(os.path.join(config[sub_element]['dataroot_gt'], 'target', '*.png')))    
                input_r_file_path_list,input_l_file_path_list =[],[]
            
            total_files = len(input_c_file_path_list)

            assert total_files > 0, 'Wrong Dataset Name or No Dataset Exist, Please Check!!'

            results_dir =  os.path.join('./Results',test_element, sub_element, folder_time)
            test(input_c_file_path_list,gt_file_path_list,input_r_file_path_list,input_l_file_path_list,net_config,net_weight,results_dir,net_mode)
    
    else:
        
        input_c_file_path_list = natsorted(glob(os.path.join(config[eval_subset]['dataroot_lq'], 'input', '*.png')))
        gt_file_path_list = natsorted(glob(os.path.join(config[eval_subset]['dataroot_gt'], 'target', '*.png')))    
        if len(input_c_file_path_list) != len(gt_file_path_list):
            gt_file_path_list = make_lf_aif_gt_dataset(input_c_file_path_list,os.path.join(config[eval_subset]['dataroot_gt'], 'target'))

        input_r_file_path_list,input_l_file_path_list =[],[]
        results_dir =  os.path.join('./Results',test_element,eval_subset,folder_time)

        total_files = len(input_c_file_path_list)
        assert total_files > 0, 'Wrong Dataset Name or No Dataset Exist, Please Check!!'

        test(input_c_file_path_list,gt_file_path_list,input_r_file_path_list,input_l_file_path_list,net_config,net_weight,results_dir,net_mode)







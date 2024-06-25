import sys
# Standard library imports
from datetime import datetime
import os
import glob
# Data handling and numerical operations
import numpy as np
import pandas as pd
# Image processing and visualization
import matplotlib.pyplot as plt
import tifffile as tiff
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Machine learning and deep learning frameworks
import torch
# Utilities for processing
from tqdm import tqdm
import re
from multiprocessing import Pool, set_start_method
from sklearn.metrics import mean_squared_error
from math import sqrt
import logging

logging.basicConfig(filename='resultslog.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


pressure_levels = [
        '10000 Pa',
        '12500 Pa', '15000 Pa', '17500 Pa', '20000 Pa', 
        '22500 Pa', '25000 Pa', '27500 Pa', '30000 Pa',
        '32500 Pa', '35000 Pa', '37500 Pa', '40000 Pa',
        '42500 Pa', '45000 Pa', '47500 Pa', '50000 Pa',
        '52500 Pa', '55000 Pa', '57500 Pa', '60000 Pa',
        '62500 Pa', '65000 Pa', '67500 Pa', '70000 Pa',
        '72500 Pa', '75000 Pa', '77500 Pa', '80000 Pa', 
        '82500 Pa', '85000 Pa', '87500 Pa', '90000 Pa',
        '92500 Pa', '95000 Pa', '97500 Pa', '100000 Pa'
                    ]


def IBTrACS_Select_TS(name, date):
    date_time_obj = datetime.strptime(date, "%Y%m%d%H")
    format_date = date_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    
    data = pd.read_csv(path,low_memory=False)
    # 只留下需要的机构数据columns
    data = data[list(filter(None,[i*(i in ['SID','SEASON', 'NAME','ISO_TIME', 'LON', 'LAT']) for i in data.columns]))]
    data = data[[i == name for i in data['NAME']]]
    data = data[[i == format_date for i in data['ISO_TIME']]]
    return data


def latlon_to_index(lat, lon, lat_original, lon_original):
    # Check if original data uses 0-360 instead of -180 to 180 for longitude
    if np.max(lon_original) > 180:
        # Adjust input longitude to 0-360 if it is negative
        if lon < 0:
            lon += 360
    # print(lat, lon, lat_original, lon_original)
    # Calculate the index for the closest match in lat_original and lon_original
    # Find the index where the difference between the given lat, lon and the array values is smallest
    lat_idx = int((lat - np.min(lat_original)) / ((np.max(lat_original) - np.min(lat_original)) / 600))
    lon_idx = int((lon - np.min(lon_original)) / ((np.max(lon_original) - np.min(lon_original)) / 600))
    # print(lat_original, lat)
    # print('latlon index', lat_idx, lon_idx)
    return lat_idx, lon_idx


def draw_3D(all_processed_images, image_original, mask, vmax_tif, save_path, dpi=300):
    fig = plt.figure(figsize=(22, 8))
    # plt.subplots_adjust(wspace=0.03)  # Adjust the width between subplots
    for i in range(1, 4):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        for j, pressure in enumerate(pressure_levels[:]):
            # j=25
            pressure = int(pressure[:-3])/100
            # 获取当前气压层的风速数据
            if i == 2:
                title = f'(b) Model Reconstruction'
                wind_speed = all_processed_images[:, :, j]
            elif i == 3:
                title = f'(c) HWRF Wind Speeds'
                wind_speed = image_original[:, :, j]
            elif i == 1:
                title = f'(a) Simulated Dropsonde'
                wind_speed = image_original[:, :, j]
            x, y = np.meshgrid(np.arange(wind_speed.shape[1]), np.arange(wind_speed.shape[0]))
            # 将x, y转换为浮点型以支持NaN
            x, y = x.astype(float), y.astype(float)
            # 设置Z坐标为当前气压值
            z = np.full(wind_speed.shape, pressure).astype(float)  # 确保z也是浮点型
            if i != 1:
                if 600 < pressure < 800:
                    x[:, :] = np.nan
                    y[:, :] = np.nan
                    z[:, :] = np.nan
                # 在指定的气压层隐藏一部分数据
                if True:
                    x[:200, 200:400] = np.nan # 第一个数字是y(Latitude)轴
            x, y, z = x.flatten(), y.flatten(), np.full(x.shape, pressure)
            c = wind_speed.flatten()  # Color based on wind speed
            if i == 1:
                c[mask.cpu().flatten() == 1] = np.nan  # 将遮罩中为1的位置设置为NaN
            # Scatter plot
            scatter = ax.scatter(x, y, z, c=c, cmap='rainbow', marker='.', vmin=0, vmax=vmax_tif)
            ax.set_title(title, fontsize=16, pad=-10)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Pressure (hPa)')
            ax.set_zlim(1000, 100)
            # plt.tight_layout()

            
    # Create a separate axis for the colorbar
    cbar_width = 0.12  # 颜色条的宽度
    cbar_left = (1 - cbar_width) / 2  # 计算颜色条左侧位置
    cbar_ax = fig.add_axes([cbar_left, 0.13, cbar_width, 0.015])  # [left, bottom, width, height]
    # Add colorbar to the separate axis
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Wind Speed (m/s)')
    # plt.subplots_adjust(wspace=0.2)
    plt.text(244, 7, 'yy')
    plt.savefig(f'{save_path}', dpi=dpi, bbox_inches='tight') # 左、右、下、上
    plt.close()
    

def draw_2D(all_processed_images, image_original, mask, vmax_tif, lat_original, lon_original, save_path, dpi=300):    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=dpi)

    for ax in axs:
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--', crs=ccrs.PlateCarree())
        gl.top_labels = False
        gl.right_labels = False

    cmap = 'rainbow'
    # vmax_tif = 70
    # 绘制原始HWRF数据
    channel_index = -7
    axs[2].pcolormesh(lon_original[..., 0], lat_original[..., 0], image_original[:, :, channel_index], cmap=cmap, vmin=0, vmax=vmax_tif, transform=ccrs.PlateCarree())
    axs[2].set_title('(c) 850hPa HWRF Wind Speeds', fontsize=16)

    # 绘制处理后的HWRF数据
    axs[1].pcolormesh(lon_original[..., 0], lat_original[..., 0], all_processed_images[:, :, channel_index], cmap=cmap, vmin=0, vmax=vmax_tif, transform=ccrs.PlateCarree())
    axs[1].set_title('(b) 850hPa Model Reconstruction', fontsize=16)

    # 绘制应用掩码的HWRF数据
    values = image_original[:, :, channel_index].copy()
    values[mask[0].cpu() == 1] = np.nan
    non_zero_indices = np.nonzero(image_original[:, :, channel_index])
    lon_coords = lon_original[non_zero_indices]
    lat_coords = lat_original[non_zero_indices]
    values = values[non_zero_indices]
    scatter = axs[0].scatter(lon_coords, lat_coords, c=values, cmap=cmap, vmin=0, vmax=vmax_tif, s=10, transform=ccrs.PlateCarree())
    axs[0].set_title('(a) 850hPa Simulated Dropsonde', fontsize=16)

    # Create a separate axis for the colorbar
    cbar_width = 0.12  # 颜色条的宽度
    cbar_left = (1 - cbar_width) / 2  # 计算颜色条左侧位置
    cbar_ax = fig.add_axes([cbar_left, 0.05, cbar_width, 0.015])  # [left, bottom, width, height]

    # Add colorbar to the separate axis
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Wind Speed (m/s)')
    # 添加颜色条
    # fig.colorbar(axs[2].collections[0], ax=axs.ravel().tolist(), orientation='horizontal', fraction=0.02, pad=0.1, aspect=50)
    plt.savefig(f'{save_path}', dpi=dpi)
    plt.close()
    

def draw_vertical(all_processed_images, image_original, mask, vmax_tif, lat_original, lon_original, save_path, dpi=300):    
    # Assuming 'vertical', 'all_processed_images' and 'pressure_levels' are defined elsewhere
    horizontal_index = 200
    start_index = 4  # 第四个值对应索引为3
    step = 8
    # Get the shape of the image
    height, width, pressure_layers = image_original.shape

    # Plot the horizontal line
    plt.figure(figsize=(20, 6), dpi=dpi)

    # Plot for image_original
    plt.subplot(1, 3, 2)
    plt.imshow(np.transpose(image_original[horizontal_index, :, :]), aspect='auto', cmap='rainbow', vmin=0, vmax=vmax_tif)
    plt.colorbar(label='Wind Speed (m/s)')
    plt.xlabel('Longitude')
    plt.ylabel('Pressure (hPa)')
    plt.yticks(np.arange(start_index, len(pressure_levels), step), [int(int(p.split()[0])/100) for p in pressure_levels[start_index::step]])  # 设置纵轴刻度
    plt.title('(b) HWRF vertical cross section')

    # Plot for all_processed_images
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(all_processed_images[horizontal_index, :, :]), aspect='auto', cmap='rainbow', vmin=0, vmax=vmax_tif)
    plt.colorbar(label='Wind Speed (m/s)')
    plt.xlabel('Longitude')
    plt.ylabel('Pressure (hPa)')
    plt.yticks(np.arange(start_index, len(pressure_levels), step), [int(int(p.split()[0])/100) for p in pressure_levels[start_index::step]])  # 设置纵轴刻度
    plt.title('(a) Reconstruction vertical cross section')

    # Plot for RMSE
    plt.subplot(1, 3, 3)
    plt.imshow(np.transpose(image_original[horizontal_index, :, :] - all_processed_images[horizontal_index, :, :]), aspect='auto')  # Adjust colormap as necessary
    plt.colorbar(label='Bias (m/s)')
    plt.xlabel('Longitude')
    plt.ylabel('Pressure (hPa)')
    plt.yticks(np.arange(start_index, len(pressure_levels), step), [int(int(p.split()[0])/100) for p in pressure_levels[start_index::step]])  # 设置纵轴刻度
    # plt.ylim(1000, 100)
    plt.title('(c) Bias')

    plt.tight_layout()
    # plt.savefig('calvin_vertival.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.savefig(f'{save_path}', dpi=dpi, bbox_inches='tight')
    plt.close()
    
        
# ori_paths = np.sort(glob.glob('/media/hanxinhai/47a2942c-0b92-4be8-a73a-b2618fd736e9/3D_recons/HWRF_all_layers/*'))
ori_paths = val_paths
mask_path = mask_path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sd_path= pth_path
generator = G().to(device)

# biparjoy02a.2023061406.hwrfprs.core.0p015.f000.grb2.tif


def process_image(ori_path):
    logging.info(f"Starting processing for {ori_path}")
    
    filename = ori_path.split('/')[-1]
    # save_path_2D = f'./2Dresults_60/{filename}.png'
    # save_path_3D = f'./3Dresults_60/{filename}.png'
    # save_path_vertical = f'./vertical_results_60/{filename}.png'
    
    save_path_2D = f'./final_results/{filename}_2D.png'
    save_path_3D = f'./final_results/{filename}_3D.png'
    save_path_vertical = f'./final_results/{filename}_vertical.png'
    # if os.path.exists(save_path_2D) and  os.path.exists(save_path_3D) and os.path.exists(save_path_vertical):
    #     print('ALL EXISTS', save_path_2D)
    #     return
    
    try:
        lon_path = lon_path
        lat_path = lat_path
        vmax_tif = 0
        processed_results = []
        rmses = []
        date = ori_path.split('.')[1]
        name = re.sub(r'\d.*$', '', ori_path.split('.')[0].split('/')[-1]).upper()
        lat, lon = np.float32(IBTrACS_Select_TS(name, date)[['LAT', 'LON']].values[0])
        print(date, name, lat, lon)
        
            # Skip processing if the cyclone name is 'INVEST'
        # if name == 'INVEST':
        #     print(f"Skipping processing for {name}")
        #     return
        
        image_original = tiff.imread(ori_path)[:600, :600, :] # Adjust as needed
        lat_original = np.expand_dims(tiff.imread(lat_path), -1)[:600, :600, :]
        lon_original = np.expand_dims(tiff.imread(lon_path), -1)[:600, :600, :]
        
        lat_idx, lon_idx = latlon_to_index(lat, lon, lat_original[:,:,0], lon_original[:,:,0])
        
        # Ensure the selected region remains within the original image boundaries
        size_start_row = max(0, lat_idx - 200)
        size_end_row = min(600, lat_idx + 200)
        size_start_col = max(0, lon_idx - 200)
        size_end_col = min(600, lon_idx + 200)
        
        # Extract the 400x400 region
        image_original = image_original[size_start_row:size_end_row, size_start_col:size_end_col, :]
        lat_original = lat_original[size_start_row:size_end_row, size_start_col:size_end_col, :]
        lon_original = lon_original[size_start_row:size_end_row, size_start_col:size_end_col, :]
        
        mask_pil = tiff.imread(mask_path)  # Ensure mask_path is defined
        mask = torch.tensor(np.array(mask_pil).transpose(2, 0, 1)).to(device)[0:1, :, :]
        all_processed_images = G(image_tensor, mask)
        
        rmse = sqrt(mean_squared_error(image_original.flatten(), all_processed_images.flatten()))
        print('RMSE:', rmse)
        rmses.append(rmse)

        # draw_2D(all_processed_images, image_original, mask, vmax_tif, lat_original, lon_original, save_path=save_path_2D, dpi=300)
        # draw_3D(all_processed_images, image_original, mask, vmax_tif, save_path=save_path_3D, dpi=300)
        # draw_vertical(all_processed_images, image_original, mask, vmax_tif, lat_original, lon_original, save_path_vertical, dpi=300)

        logging.info(f"Finished processing for {ori_path}")
        return rmse
    except Exception as e:
        print(e)
        logging.error(f"Error processing {ori_path}: {e}", exc_info=True)


if __name__ == '__main__':
    # Set the start method for multiprocessing to 'spawn'
    # set_start_method('spawn', force=True)
    
    # with Pool(processes=10) as pool:  # Adjust number of processes as needed
    #     pool.map(process_image, ori_paths)
    #     pool.close()
    #     pool.join()
    
    set_start_method('spawn', force=True)  # Set the start method for multiprocessing
    rmse_lst = []
    with Pool(processes=10) as pool:
        rmses = pool.map(process_image, ori_paths)
        rmses = [rmse for rmse in rmses if rmse is not None]
        # print('rmses', rmses)
        pool.close()
        pool.join()

        # Remove None values from results due to failed processing
        
        # rmses = [rmse for rmse in rmses if rmse is not None]
    
    # Calculate and log the average RMSE
    if rmses:
        average_rmse = sum(rmses) / len(rmses)
        print(mask_path, 'RMSE AVG:', average_rmse)
        logging.info(f"Average RMSE across all processed images: {average_rmse}")
    else:
        logging.info("No RMSE values calculated due to processing failures.")
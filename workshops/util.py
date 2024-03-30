import cv2
from matplotlib import pyplot as plt
import numpy as np

# cv2.IMREAD_COLOR : Carga la imagen a color, omitiendo transparencias. Es la bandera por defecto.
# cv2.IMREAD_GRAYSCALE : Carga la imagen en escala de grises.
# cv2.IMREAD_UNCHANGED : Carga la imagen como tal, incluyendo el canal alpha si existe.

#  -------------------- COLOR SPACES -------------------- 

def image_read(
        image_path: str, 
        mode: str = 'color'
    ) -> np.ndarray:
    try:
        grayscale_modes = ('grayscale', 'greyscale', 'gray', 'grey', 'gris')
        mode = mode.lower()
        if mode == 'color':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        elif mode == 'standard-color':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        elif mode in grayscale_modes:
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        elif mode == 'yuv':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2YUV)
        elif mode == 'hsv':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2HSV)
        elif mode == 'hls':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2HLS)
        elif mode == 'lab':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2LAB)
        elif mode == 'xyz':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2XYZ)
        else:
            raise ValueError(f'Invalid mode. Use "color", "standard-color", "{grayscale_modes}", "yuv", "hsv", "hls", "lab" or "xyz".')
    except Exception as e:
        print(f'Error: {e}')

def image_show(
        image: np.ndarray, 
        title: str = 'Image', 
        figsize: tuple = None
    ) -> None:
    try:
        plt.figure(figsize = figsize)
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        plt.title(title)
        plt.show()
    except Exception as e:
        print(f'Error: {e}')

def subplot_images(
        title: str = '', 
        images: list = [],
        images_name: list = [], 
        figsize: tuple = (15, 7)
    ) -> None:
    try:
        fig, axes = plt.subplots(1, len(images), figsize = figsize)
        fig.suptitle(title, fontsize = 20)
        for ax, image, name in zip(axes, images, images_name):
            ax.set_title(name)
            ax.imshow(image, cmap='gray')
        plt.show()
    except Exception as e:
        print(f'Error: {e}')

def plot_image_and_histogram(
        image: np.ndarray, 
        title: str = '', 
        image_title: str = '', 
        hist_title: str = '', 
        figsize=(15, 4)
    ) -> None:
    try:
        if len(image.shape) == 3:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle(title, fontsize = 20)
            colors = ('r', 'g', 'b')
            axs[0].set_title(image_title)
            axs[0].imshow(image, cmap='gray')

            axs[1].set_title(hist_title)
            for i, col in enumerate(colors):
                img_array_i = image[:,:,i].ravel()
                axs[1].hist(
                    img_array_i,
                    histtype='step', 
                    bins=255, 
                    range=(0.0, 255.0), 
                    density=True, 
                    color=colors[i]
                )
            plt.show()
        else:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle(title, fontsize = 20)
            axs[0].set_title(image_title)
            axs[0].imshow(image, cmap='gray')

            axs[1].set_title(hist_title)
            axs[1].hist(
                image.ravel(), 
                bins=255, 
                range=(0.0, 255.0), 
                density=True
            )
            plt.show()
    except Exception as e:
        print(f'Error: {e}')

def plot_channels(
        image: np.ndarray,
        mode = 'rbg',
        title: str = 'RGB channels', 
        channel_names: tuple = ['Channel R', 'Channel G', 'Channel B'],
        cmaps: tuple = ('Reds', 'Greens', 'Blues'),
        figsize: tuple = (30, 7),
        images: list = []
    ) -> None:
    # Split the image into its channels
    try:
        mode = mode.lower()    
        if mode == 'rgb':
            channels = [image[:,:,i] for i in range(3)]
        elif mode == 'bgr':
            channels = [image[:,:,i] for i in [2, 1, 0]]
            cmaps = cmaps[::-1]
            channel_names = channel_names[::-1]
        elif mode == 'cmy':
            channels = [255 - image[:,:,i] for i in range(3)]
            cmaps = ('GnBu', 'RdPu', 'YlOrBr')
            channel_names = ('Channel C', 'Channel M', 'Channel Y')
        elif mode == 'yiq':
            channels = [
                0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2],
                0.596 * image[:,:,0] - 0.274 * image[:,:,1] - 0.322 * image[:,:,2],
                0.211 * image[:,:,0] - 0.523 * image[:,:,1] + 0.312 * image[:,:,2]
            ]
            cmaps = ('gray', 'gray', 'gray')
            channel_names = ('Channel Y', 'Channel I', 'Channel Q')
        elif mode == 'yuv':
            channels = [image[:,:,i] for i in range(3)]
            cmaps = ('gray', 'gray', 'gray')
            channel_names = ('Channel Y', 'Channel U', 'Channel V')
        elif mode == 'hsl':
            # Convert to HSL
            size = np.shape(image)
            image_HSL =np.zeros((size), dtype=np.float32)
            # Algorithm to convert RGB to HSL
            for i in range(size[0]):
                for j in range(size[1]):
                    # Normalization
                    max_value = np.max(image[i][j])
                    min_value = np.min(image[i][j])
                    
                    channel_S = max_value - min_value
                    channel_L = channel_S / 2
                    
                    image_HSL[i][j][1] = channel_S
                    image_HSL[i][j][2] = channel_L
                    
                    if(max_value == min_value):
                        image_HSL[i][j][0] = 0
                        continue
                    
                    red = image[i][j][0]
                    green = image[i][j][1]
                    blue = image[i][j][2]
                    
                    if(max_value == red):
                        channel_H = ( green - blue ) * 60 / ( max_value - min_value )
                    elif(max_value == green):
                        channel_H = ( blue - red ) * 60 / ( max_value - min_value ) + 120
                    else:
                        channel_H = ( red - green ) * 60 / ( max_value - min_value ) + 240
                    if channel_H >= 0:
                        image_HSL[i,j,0] = channel_H
                    else:
                        image_HSL[i,j,0] = 360.0 - channel_H
            channels = [image_HSL[:,:,i] for i in range(3)]
            cmaps = ('gray', 'gray', 'gray')
            channel_names = ['Channel H', 'Channel S', 'Channel L']
        elif mode == 'hsv':
            channels = [image[:,:,i] for i in range(3)]
            cmaps = ('gray', 'gray', 'gray')
            channel_names = ['Channel H', 'Channel S', 'Channel V']
        elif mode == 'lab':
            channels = [image[:,:,i] for i in range(3)]
            cmaps = ('gray', 'gray', 'gray')
            channel_names = ['Channel L', 'Channel A', 'Channel B']
        elif mode == 'xyz':
            channels = [image[:,:,i] for i in range(3)]
            cmaps = ('gray', 'gray', 'gray')
            channel_names = ['Channel X', 'Channel Y', 'Channel Z']
        elif mode == 'custom':
            channels = [image[:,:,i] for i in range(3)]
            cmaps = cmaps
            channel_names = channel_names
        elif mode == 'individual': 
            channels = images
            cmaps = cmaps
            channel_names = channel_names
        else:
            raise ValueError(f'Invalid mode. Use "rgb", "bgr", "cmy", "yiq", "yuv", "hsl", "hsv", "lab", "xyz", "custom" or "individual".')

        # Plot the channels
        fig, axes = plt.subplots(1, 3, figsize = figsize)
        fig.suptitle(title, fontsize = 20)

        for ax, channel, name, cmap in zip(axes, channels, channel_names, cmaps):
            ax.set_title(name)
            ax.imshow(channel, cmap=cmap, aspect='auto')
    except Exception as e:
        print(f'Error: {e}')

#  --------- BASIC PIXEL TO PIXEL TRANSFORMATIONS ---------
        
def linear_transformation(
        image: np.ndarray, 
        alpha: float, 
        beta: float, 
        mode='default'
    ) -> np.ndarray:
    try:
        img = image.copy()
        if mode == 'default':
            return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        elif mode == 'grayscale':
            return cv2.add(cv2.multiply(img, alpha), beta)
        elif mode == 'color':
            size = np.shape(img)
            for i in range(size[0]):
                for j in range(size[1]):
                    for k in range(size[2]):
                        value = img[i][j][k] * alpha + beta
                        if(value > 255):
                            img[i][j][k] = 255
                        elif(value < 0):
                            img[i][j][k] = 0
                        else:
                            img[i][j][k] = value
            return img
        else:
            raise ValueError(f'Invalid mode. Use "default", "grayscale" or "color".')
    except Exception as e:
        print(f'Error: {e}')

def negative_transformation(image: np.ndarray) -> np.ndarray:
    try:
        return linear_transformation(image, -1, 255)
    except Exception as e:
        print(f'Error: {e}')

#  -------------------- IMAGE TRANSFORMATIONS --------------------
        
def apply_transformation_on_rgb(
        image: np.ndarray, 
        transformation: callable, args: list
    ) -> np.ndarray:
    try:
        image_transformated = np.zeros(image.shape, np.uint8)
        for i in range(3):
            image_transformated[:,:,i] = transformation(image[:,:,i], *args)
        return image_transformated
    except Exception as e:  
        print(f'Error: {e}')

def gamma_correction(
        image: np.ndarray, 
        a: int, 
        gamma: float
    ) -> np.ndarray:
    try:
        image_result = cv2.multiply(cv2.pow(image.copy().astype(np.float32) / 255.0, gamma), a)
        return np.clip(image_result * 255.0, 0, 255).astype(np.uint8)
    except Exception as e: 
        print(f'Error: {e}')

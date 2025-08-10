import pygame
import sounddevice as sd
import numpy as np
import torch
import random
import os
from scipy.fft import rfft
from dcgan import GrayScaleGenerator
from dcgan import ColorGenerator

from torch.autograd import Variable

from graph import draw_line
from graph import Knob
from recording import get_sample


height, width = 250, 500

pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

running = True

rate = 44100
blocksize = 500

device = 'default'

stream = sd.InputStream(
    samplerate=rate,
    blocksize=blocksize,
    channels=1,
    dtype='float32',
    device=device
)

latent_dim= 100
image_size = 64
image_channels = 1
model_path_grayscale = "gan/cifar100_dcgan_grayscale/weights/netG_epoch_299.pth"
model_path = "eyes/andyG-epoch300.pth"

#generator = ColorGenerator(image_size, latent_dim, image_channels)
generator = GrayScaleGenerator(0)
generator.load_state_dict(torch.load(model_path_grayscale, map_location=torch.device('cpu')))
generator.eval()

smoothing_factor = 0.6
noise_weigth = 0.3
audio_weigth = 0.3

smoothing_factor_knob = Knob(center=(30,30), radius=20, start_value=smoothing_factor)
noise_weigth_knob = Knob(center=(80,30), radius=20, start_value=noise_weigth)
audio_weigth_knob = Knob(center=(130,30), radius=20, start_value=audio_weigth)


with stream:
    step = 0
    a = torch.randn(1, latent_dim)
    b = torch.randn(1, latent_dim)

    lookup = np.arange(latent_dim)
    #random.shuffle(lookup)
    smoothed_spectrum = np.zeros(int(blocksize/2) + 1)

    while running:
        step += 1
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
            smoothing_factor_knob.handle_event(ev)
            noise_weigth_knob.handle_event(ev)
            audio_weigth_knob.handle_event(ev)

        smoothing_factor = smoothing_factor_knob.value
        noise_weigth = noise_weigth_knob.value
        audio_weigth = audio_weigth_knob.value

        # Audio Stream Input
        smoothed_spectrum = get_sample(stream, smoothed_spectrum, blocksize, smoothing_factor)

        # Noise Transform
        if step % 10 == 0:
            c = random.randint(0, latent_dim - 1)
            d = random.randint(0, latent_dim - 1)
            lookup[d], lookup[c] = lookup[c], lookup[d]
        if step % 20 == 0:
            a[0][random.randint(0, latent_dim - 1)] = torch.randn(1)[0]
            b[0][random.randint(0, latent_dim - 1)] = torch.randn(1)[0]

        spectrum = np.zeros(latent_dim)
        for i in range(latent_dim):
            spectrum[i] = smoothed_spectrum[lookup[i]]

        # Image Generation
        noise = torch.zeros(1, latent_dim)
        for i in range(latent_dim):
            noise[0][i] = a[0][i] * noise_weigth + spectrum[i] * audio_weigth * b[0][i]


        noise = noise.view(1,100,1,1)
        image = generator(noise).detach().squeeze()
        #fake_image = torch.zeros((3,32,32))
        image_array = image.numpy()
        image_array = ((image_array + 1) / 2.0 * 255).astype(np.uint8) # -1,1 -> 0,255

        if image_channels == 1:
            image_array = np.stack([image_array]*3, axis=0)

        image_rgb = np.transpose(image_array, (1,2,0))
        surface = pygame.surfarray.make_surface(np.transpose(image_rgb, (1, 0, 2)))
        
        windows = 2

        surface = pygame.transform.smoothscale(surface, (width / windows, height ))  # scale up for display
        

        screen.fill((255, 255, 255))

        for i in range(windows):
            #flipped_surface = pygame.transform.flip(surface, i % 2 == 0, False)
            screen.blit(surface, ((width / windows) * i, (height / windows) * 0))

        path = "/tmp/current.png"
        tmp_path = path + ".tmp"
        pygame.image.save(surface, tmp_path)
        os.replace(tmp_path, path)
        draw_line(screen, smoothed_spectrum)

        smoothing_factor_knob.draw(screen)
        noise_weigth_knob.draw(screen)
        audio_weigth_knob.draw(screen)

        pygame.display.flip()
        clock.tick(60)

pygame.quit()

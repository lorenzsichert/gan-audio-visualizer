import numpy as np
import pygame
import pygame.gfxdraw
import math


def draw_line(screen, smoothed_spectrum):
    width, height = screen.get_width(), screen.get_height()
    points = []

    x = 0
    for i in smoothed_spectrum:
        x += 1
        points.append(((float(x) / len(smoothed_spectrum)) * width, height - np.sqrt(float(i)) * 10))

    pygame.draw.lines(screen, (255, 255, 255), False, points, 2)

class Knob:
    def __init__(self, center, radius, min_value=0, max_value=1, start_value=0.5, sensitivity=0.002):
        self.center = center
        self.radius = radius
        self.min_value = min_value
        self.max_value = max_value
        self.value = start_value
        self.sensitivity = sensitivity

        self.dragging = False
        self.last_mouse_x = None

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check if click is inside knob
            if (event.pos[0] - self.center[0]) ** 2 + (event.pos[1] - self.center[1]) ** 2 <= self.radius ** 2:
                self.dragging = True
                self.last_mouse_x = event.pos[0]

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            dx = event.pos[0] - self.last_mouse_x
            self.value += dx * self.sensitivity * (self.max_value - self.min_value)
            self.value = max(self.min_value, min(self.max_value, self.value))
            self.last_mouse_x = event.pos[0]

    def draw(self, surface):
        import math
        # Draw knob base
        pygame.gfxdraw.aacircle(surface, self.center[0], self.center[1], self.radius, (100, 100, 100))
        pygame.gfxdraw.filled_circle(surface, self.center[0], self.center[1], self.radius, (100, 100, 100))
        pygame.gfxdraw.aacircle(surface, self.center[0], self.center[1], self.radius - 3 , (50, 50, 50))
        pygame.gfxdraw.filled_circle(surface, self.center[0], self.center[1], self.radius - 3, (50, 50, 50))

        # Map value to angle
        angle_range = 360
        start_angle = -90
        value_norm = (self.value - self.min_value) / (self.max_value - self.min_value)
        angle = start_angle + value_norm * angle_range

        # Draw indicator line
        line_length = self.radius * 0.8
        end_x = self.center[0] + line_length * math.cos(math.radians(angle))
        end_y = self.center[1] + line_length * math.sin(math.radians(angle))
        pygame.draw.aaline(surface, (200,200,200), self.center, (end_x, end_y), 1)

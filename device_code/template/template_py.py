
from __future__ import annotations
from sanajeh import DeviceAllocator
import os, sys, time, pygame, random
kSeed: int = 43

class Renderer():

    def __init__(self, width, height):
        self.width = width
        self.height = height
        os.environ['SDL_VIDEODRIVER'] = 'x11'
        pygame.init()
        self.window = pygame.display.set_mode(((self.width * 6), (self.height * 6)))
        self.screen = pygame.Surface((self.width, self.height))
        self.screen.fill((0, 0, 0))
        self.pxarray = pygame.PixelArray(self.screen)

    def update(self):
        self.window.blit(pygame.transform.scale(self.screen, self.window.get_rect().size), (0, 0))
        pygame.display.update()

    def setpixel(self, x, y, r, g, b):
        self.pxarray[(x, y)] = pygame.Color(r, g, b)

    def initscreen(self):
        self.screen.fill((0, 0, 0))

class SampleClass():

    def __init__(self):
        self.random_state_: DeviceAllocator.RandomState = None
        self.x: int = 0
        self.y: int = 0

    def SampleClass(self, seed: int):
        random.seed(seed)
        self.x = (random.getrandbits(32) % 50)
        self.y = (random.getrandbits(32) % 50)

    def update(self):
        rand1: int = ((random.getrandbits(32) % 3) - 1)
        rand2: int = ((random.getrandbits(32) % 3) - 1)
        self.x += rand1
        self.y += rand2

def main(allocator, do_render):
    width = 100
    height = 100
    num = 100
    iterate = 100
    render = Renderer(width, height)
    render.update()
    allocator.initialize()
    allocator.parallel_new(SampleClass, num)

    def do_render(tmp):
        render.setpixel(tmp.x, tmp.y, 255, 255, 255)
    for i in range(iterate):
        allocator.parallel_do(SampleClass, SampleClass.update)
        allocator.do_all(SampleClass, do_render)
        render.update()
        render.initscreen()

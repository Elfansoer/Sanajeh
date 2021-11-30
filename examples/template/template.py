from __future__ import annotations
from sanajeh import DeviceAllocator

import os, sys, time, pygame, random

kSeed: int = 43 # this is hardcoded and must exist

# Utility function causes error about superclass ():
# def randrange(r: int):
#   return random.getrandbits(32)%r
# def delete_cell(cell: Cell):
#     DeviceAllocator.destroy(cell)
#     cell = None

# Renderer class
class Renderer:
  def __init__(self,width,height):
    self.width = width
    self.height = height

    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.init()
    self.window = pygame.display.set_mode((self.width*6, self.height*6))
    self.screen = pygame.Surface((self.width, self.height))
    self.screen.fill((0,0,0))
    self.pxarray = pygame.PixelArray(self.screen)

  def update(self):
    self.window.blit(pygame.transform.scale(self.screen, self.window.get_rect().size), (0,0))
    pygame.display.update()

  def setpixel(self,x,y,r,g,b):
    self.pxarray[x,y] = pygame.Color(r,g,b)

  def initscreen(self):
    self.screen.fill((0,0,0))

# Cuda Classes
class SampleClass:
  def __init__(self):
    self.x: int = 0
    self.y: int = 0

  def SampleClass(self,seed: int):
    random.seed(seed) # this is hardcoded and must exist
    self.x = random.getrandbits(32)%50
    self.y = random.getrandbits(32)%50

  def update(self):
    rand1: int = random.getrandbits(32)%3 - 1
    rand2: int = random.getrandbits(32)%3 - 1
    self.x += rand1
    self.y += rand2

# Main function
def main(allocator, do_render):
  width = 100
  height = 100
  num = 100
  iterate = 100

  # init renderer
  render = Renderer(width,height)
  render.update()

  # init allocator
  allocator.initialize()

  # create objects
  allocator.parallel_new(SampleClass, num)

  # do_all function
  def do_render(tmp):
    render.setpixel(tmp.x,tmp.y,255,255,255)
  
  for i in range(iterate):

    # parallel do
    allocator.parallel_do(SampleClass, SampleClass.update)

    # for rendering, do all
    allocator.do_all(SampleClass, do_render)

    # rendering update
    render.update()
    render.initscreen()
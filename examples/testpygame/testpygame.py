from __future__ import annotations
from sanajeh import DeviceAllocator

import os, sys, time, pygame, random

kSeed: int = 43 # this is hardcoded and must exist
kSizeX: int = 100
kSizeY: int = 100
kNum: int = 20
kNumIterations: int = 500

# Utility function causes error about superclass ():
# def randrange(r: int):
#   return random.getrandbits(32)%r
# def delete_cell(cell: Cell):
#     DeviceAllocator.destroy(cell)
#     cell = None

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

class Cell:
  def __init__(self):
    self.x: int = 0
    self.y: int = 0
    self.r: int = 0
    self.g: int = 0
    self.b: int = 0
    self.life: int = 0
    self.birth: int = 0

  def Cell(self,seed: int):
    random.seed(seed) # this is hardcoded and must exist
    self.x = random.getrandbits(32)%kSizeX
    self.y = random.getrandbits(32)%kSizeY
    self.r = random.getrandbits(32)%255
    self.g = random.getrandbits(32)%255
    self.b = random.getrandbits(32)%255
    self.life = 20
    self.birth = 10

  def update(self):
    rand: int = random.getrandbits(32)%4
    if rand==0:
      self.x = (self.x+1)%kSizeX
    if rand==1:
      self.x = (self.x-1)%kSizeX
    if rand==2:
      self.y = (self.y+1)%kSizeY
    if rand==3:
      self.y = (self.y+1)%kSizeY

    self.birth -= 1
    self.life -= 1
    if self.birth==0:
        self.birth = 10
        new_cell: Cell = DeviceAllocator.new(Cell, random.getrandbits(32))
        new_cell.x = self.x
        new_cell.y = self.y

    if self.life==0:
      DeviceAllocator.destroy(self)

def main(allocator, do_render):
  render = Renderer(kSizeX,kSizeY)
  render.update()

  def do_render(cell):
    render.setpixel(cell.x,cell.y,cell.r,cell.g,cell.b)

  allocator.initialize()
  allocator.parallel_new(Cell, kNum)
  
  for i in range(kNumIterations):
    allocator.parallel_do(Cell, Cell.update)

    allocator.do_all(Cell, do_render)
    render.update()
    render.initscreen()
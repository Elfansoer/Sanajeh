from __future__ import annotations

import os, math, time, random
import pygame
from sanajeh import DeviceAllocator

class Projectile:
  def __init__(self):
    self.x: float = 10
    self.y: float = 10
  def Projectile(self,idx: int):
    pass
  def update(self):
    self.x += 10
    self.y += 10

def main(allocator, do_render):
  """
  Rendering setting
  """
  def render(b):
    px = int((b.pos.x + 1) * 150)
    py = int((b.pos.y + 1) * 150)
    pygame.draw.circle(screen, (255, 255, 255), (px, py), b.mass/10000*20)

  if (do_render):
    os.environ["SDL_VIDEODRIVER"] = "windib"
    screen_width = 300
    screen_height = 300
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.flip()  

  num: int = 10
  iter: int = 5000

  allocator.initialize()
  initialize_time = time.perf_counter()
  allocator.parallel_new(Projectile, num)
  parallel_new_time = time.perf_counter()

  for x in range(iter):
      allocator.parallel_do(Projectile, Projectile.update)
      # if (do_render):
      #   allocator.do_all(Body, render)
      #   pygame.display.flip()
      #   screen.fill((0, 0, 0))  
      end_time = time.perf_counter()

  print("parallel new time(%-5d objects): %.dµs" % (num, ((parallel_new_time - initialize_time) * 1000000)))
  print("average computation time: %dµs" % ((end_time - parallel_new_time) * 1000000 / iter))
  print("overall computation time(%-4d iterations): %dµs" % (iter, ((end_time - parallel_new_time) * 1000000)))  



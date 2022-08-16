import pygame as pg
from pygame import init, font, display as dp, time, event, mouse as ms
import numpy as np
from torchvision.transforms import ToPILImage
import sys
from matplotlib.pyplot import sca
from pygame import display as dp
import numpy as np
import pygame_textinput
import os
from variables import white, black, red, fps, w, h, pw, ph, pws, phs

input = pygame_textinput.TextInputVisualizer(font_color=red, cursor_color=red)

def prepareDir():
  global path
  global count
  dirName = input.value if input.value != '' else 'noLabel'
  path = f'./own/{dirName}'
  if not os.path.exists(path): os.makedirs(path)
  count = len(next(os.walk(path))[2])

def saveImg(img):
  global count
  prepareDir()
  img.save(f'{path}/{count}.png')
  count = len(next(os.walk(path))[2])

sc = dp.set_mode((w, h))
grid = np.zeros((ph, pw))
prepareDir()

run = True
drawing = False

init()
font.init()
clock = time.Clock()

def draw(pos):
  x, y = pos
  row, col = y // phs, x // pws
  grid[row, col] = white
  grid[row - 1, col] = white
  grid[row, col - 1] = white

def drawGrid():
  sc.fill(black)
  for i, row in enumerate(grid):
    for j, pixel in enumerate(row):
      pg.draw.rect(sc, (pixel, pixel, pixel), (j * pws, i * phs, pws, phs))

  sc.blit(input.surface, (10, 10))
  sc.blit(font.SysFont("comicsans", 40).render(f'{count}', 0, red), (w - 30, 10))
 

while True:
  drawGrid()

  events = event.get()
  input.update(events)

  for e in events:
    match e.type:
      case pg.QUIT: sys.exit()
      case pg.KEYDOWN:
        match e.key:
          case pg.K_RETURN: prepareDir()
      case pg.MOUSEBUTTONDOWN:
        match e.button:
          case 1: drawing = True
          case 3: grid.fill(black)
          case 2: 
            imgData = grid.reshape((pw, ph, 1)).astype(np.uint8)
            img = ToPILImage()(imgData)
            saveImg(img)
            grid.fill(black)
      case pg.MOUSEBUTTONUP:
        match e.button:
          case 1: drawing = False
          
  if drawing: draw(ms.get_pos())

  dp.update()
  clock.tick(fps)

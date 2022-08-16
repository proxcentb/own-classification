import torch
import pygame as pg
from pygame import init, font, display as dp, time, event, mouse as ms
import numpy as np
import sys
from pygame import display as dp
import numpy as np
from variables import white, black, fps, w, h, pw, ph, pws, phs, device, trainPath
from torchvision.datasets import ImageFolder

model = torch.load("model.pth")
model.eval()

sc = dp.set_mode((w, h))
grid = np.zeros((ph, pw))
labels = ImageFolder(root=trainPath).class_to_idx
inv_labels = {v: k for k, v in labels.items()}

run = True
drawing = False

init()
font.init()
clock = time.Clock()

def formatWithLabels(output):
  output = torch.nn.functional.relu(output)
  # sum = torch.sum(output, dim=1)[0].item()
  # output = output / sum
  result = {}
  for i in range(0, len(inv_labels)):
      result[inv_labels[i]] = output[0, i].item()
  return torch.max(output, dim=1)[0].item(), result

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

  data = torch.from_numpy(grid).view(1, 1, pw, ph)
  max, predictions = formatWithLabels(model(data.to(device).float(), train=False))
  for i, (k, v) in enumerate(predictions.items()):
      val = round(v * 100, 2)
      text = font.SysFont("comicsans", 24).render(f'{k}: {val}%', 0, (255, 0, 0) if v == max else (255, 255, 255))
      sc.blit(text, (10, 10 + i * 24))

while True:
  drawGrid()

  events = event.get()

  for e in events:
    match e.type:
      case pg.QUIT: sys.exit()
      case pg.MOUSEBUTTONDOWN:
        match e.button:
          case 1: drawing = True
          case 3: grid.fill(black)
      case pg.MOUSEBUTTONUP:
        match e.button:
          case 1: drawing = False
          
  if drawing: draw(ms.get_pos())

  dp.update()
  clock.tick(fps)

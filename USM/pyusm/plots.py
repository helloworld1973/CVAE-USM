# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:15:53 2023

@author: Wuestney
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class cgr_plot:
    def __init__(self, coords, coord_dict):
        #self.fig = fig
        #self.ax = ax
        self.verts = len(coord_dict)
        self.points = len(coords)
        self.coord_dict = coord_dict
        #get vertex coordinates
        sensors, vertices = tuple(zip(*coord_dict.items()))
        x_vals, y_vals = tuple((zip(*vertices)))
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        self.x_vals = x_vals
        self.y_vals = y_vals
        vert_coords = np.column_stack((x_vals, y_vals))
        xmin = x_vals.min() - 0.2
        xmax = x_vals.max() + 0.2
        ymin = y_vals.min() - 0.2
        ymax = y_vals.max() + 0.2
        self.xlims = (xmin, xmax)
        self.ylims = (ymin, ymax)
        #initiate figure instance
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        self.ax.set(xlim=self.xlims, ylim=self.ylims)
        
        self.coords = coords
        
    def plot(self):
        x, y = zip(*self.coords)
        self.ax.scatter(self.x_vals, self.y_vals, c='b', s=1)
        #self.ax.legend()
        for i, xy in self.coord_dict.items():
            if xy[0] < 0 and xy[1] > 0:
                self.ax.annotate(f'{i}', xy, xycoords='data',xytext=(-70,4), textcoords='offset points', size=10)
            elif xy[0] < 0 and xy[1] < 0:
                if len(i) < 15:
                    self.ax.annotate(f'{i}', xy, xycoords='data',xytext=(-70,-4), textcoords='offset points', size=10)
                else:
                    self.ax.annotate(f'{i}', xy, xycoords='data',xytext=(-110,-4), textcoords='offset points', size=10)
            elif xy[0] > 0 and xy[1] < 0:
                self.ax.annotate(f'{i}', xy, xycoords='data',xytext=(4,-10), textcoords='offset points', size=10)
            else:
                self.ax.annotate(f'{i}', xy, xycoords='data', xytext=(4,4), textcoords='offset points', size=10)
        self.ax.scatter(x, y, s=1, c='g')
        # c = self.ax.hexbin(x, y, gridsize=120, cmap='cool', linewidths=1, mincnt=1)
        # cb = self.fig.colorbar(c, label='count in bin')
        return
        
    def savefig(self, filename, **kwargs):
        self.fig.savefig(filename, **kwargs)
        return

    def init_frame(self):
        self.ax.cla()
        self.ax.set(xlim=self.xlims, ylim=self.ylims)
        self.ax.scatter(x=0, y=0.25, s=1, c='r', label='inital point')
        self.ax.scatter(self.x_vals, self.y_vals, c='b', label='vertices')
        self.ax.legend()
        for i, xy in enumerate(zip(self.x_vals, self.y_vals)):
            self.ax.annotate(f'{i}', xy, xycoords='data', xytext=(4,4), textcoords='offset points')
        return
    
    def animation(self, i):
        i_from = i * self.chunks
        # are we on the last frame?
        if i_from + self.chunks > len(self.coords) - 1:
            i_to = len(self.coords) - 1
        else:
            i_to = i_from + self.chunks
        rows = self.coords[i_from:i_to]
        x, y = zip(*rows)
        self.ax.scatter(x, y, s=1, c='g')
        return
    
    def animate(self, chunks=10):
        self.chunks = chunks
        self.frame_chunks = self.points // self.chunks
        self.ani = FuncAnimation(self.fig, self.animation, frames=self.frame_chunks, init_func=self.init_frame, interval=0.5, repeat=True, blit=True)
        plt.show()
        
    def movie(self):
        movie1 = self.ani.to_jshtml()
        return movie1
    
    def pause(self):
        self.ani.pause()
        return
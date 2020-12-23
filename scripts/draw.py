# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:55:31 2020

@author: lucas
"""
import cv2
import numpy as np
from win32api import GetSystemMetrics
from seaborn.categorical import _BarPlotter
# ============================================================================

FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)

# ============================================================================

class PolygonDrawer(object):
    def __init__(self, window_name, img):
        self.window_name = window_name # Name for our window
        self.window_size = (GetSystemMetrics(0), GetSystemMetrics(1))
        self.img = img # image object to draw on
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        
    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            # print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            # print("Completing polygon with %d points." % len(self.points))
            self.done = True

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, np.zeros(self.img.size, np.uint8))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            # canvas = np.zeros(CANVAS_SIZE, np.uint8)
            canvas = np.copy(self.img)
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        canvas = np.copy(self.img)
        # of a filled polygon
        if (len(self.points) > 0):
            cv2.polylines(canvas, np.array([self.points]), True, FINAL_LINE_COLOR, 1)
            cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()
        cv2.destroyWindow(self.window_name)
        
        # return large_polygon.astype("int32")
        return np.array([self.points]).astype("int32")


def annotate(batch):
    """Annotate the plant with the accession and replicate"""
    accession = input("tell me the accession ({})".format(batch))
    replicate = input("tell me the replicate ({})".format(batch))
    confirmation = input("accession: " + accession + " replicate: " + replicate + " confirm: y/n")
    if confirmation == "y" or confirmation == "":
        print("confirmed")
        return accession, replicate
    else:
        print("not confirmed, enter again")
        annotate(batch)


class _StackBarPlotter(_BarPlotter):
    """ Stacked Bar Plotter
 
    A modification of the :mod:`seaborn._BarPlotter` object with the added ability of
    stacking bars either verticaly or horizontally. It takes the same arguments
    as :mod:`seaborn._BarPlotter` plus the following:
 
    Arguments
    ---------
    stack : bool
        Stack bars if true, otherwise returns equivalent barplot as
        :mod:`seaborn._BarPlotter`.
    """
    def draw_bars(self, ax, kws):
        """Draw the bars onto `ax`."""
        # Get the right matplotlib function depending on the orientation
        barfunc = ax.bar if self.orient == "v" else ax.barh
        barpos = np.arange(len(self.statistic))
        
        if self.plot_hues is None:
            
            # Draw the bars
            barfunc(barpos, self.statistic, self.width,
                    color=self.colors, align="center", **kws)

            # Draw the confidence intervals
            errcolors = [self.errcolor] * len(barpos)
            self.draw_confints(ax,
                            barpos,
                            self.confint,
                            errcolors,
                            self.errwidth,
                            self.capsize)
        else:
            # Stack by hue
            for j, hue_level in enumerate(self.hue_names):

                barpos_prior = None if j == 0 else np.sum(self.statistic[:, :j], axis=1)

                # Draw the bars
                if self.orient == "v":
                    barfunc(barpos, self.statistic[:, j], self.nested_width,
                            bottom=barpos_prior, color=self.colors[j], align="center",
                            label=hue_level, **kws)
                elif self.orient == "h":
                    barfunc(barpos, self.statistic[:, j], self.nested_width,
                            left=barpos_prior, color=self.colors[j], align="center",
                            label=hue_level, **kws)

                # Draw the confidence intervals
                if self.confint.size:
                    confint = self.confint[:, j] if j == 0 else np.sum(self.confint[:, :j], axis=1)
                    errcolors = [self.errcolor] * len(barpos)
                    self.draw_confints(ax,
                                    barpos,
                                    confint,
                                    errcolors,
                                    self.errwidth,
                                    self.capsize)

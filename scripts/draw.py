# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:55:31 2020

@author: lucas
"""
import cv2
import numpy as np
from win32api import GetSystemMetrics

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
        
        self.scale_f = 1 #self.get_scale_f()
        self.img_small = self.img #self.downsize_img()

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True

    def get_scale_f(self):
        # Find downscale / upscale so img fits window
        resize_x = self.window_size[0] / self.img.shape[0]
        resize_y = self.window_size[1] / self.img.shape[1]
        scale_f = min(resize_x, resize_y) * .9
        return scale_f
        
    def downsize_img(self):
        new_dims = (int(self.img.shape[0] * self.scale_f), int(self.img.shape[1] * self.scale_f))
        return cv2.resize(self.img, new_dims, interpolation = cv2.INTER_CUBIC )
    
    def upsize_polygon(self):
        
        ow, oh = self.img_small.shape
        w, h = self.img.shape
        
        M = cv2.getAffineTransform(np.float32([[ow, 0], [ow, oh], [0, oh]]), np.float32([[w, 0], [w, h], [0, h]])) 
        
        points_arr = np.array([self.points]).astype("int32")
        return cv2.transform(points_arr, M)
        # import pdb; pdb.set_trace()
        # return scaled_points
        # x,y,w,h = cv2.boundingRect(points_arr)
        # center = (x + w/2, y + h/2)
        # points_arr = points_arr - center
        # points_scaled = points_arr / self.scale_f
        # return np.around(points_scaled + center).astype("int32")
        # return np.around(points_arr / self.scale_f).astype("int32")

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, np.zeros(self.img_small.size, np.uint8))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            # canvas = np.zeros(CANVAS_SIZE, np.uint8)
            canvas = np.copy(self.img_small)
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
        # canvas = np.zeros(CANVAS_SIZE, np.uint8)
        canvas = np.copy(self.img_small)
        # of a filled polygon
        if (len(self.points) > 0):
            cv2.polylines(canvas, np.array([self.points]), True, FINAL_LINE_COLOR, 1)
            cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()
        cv2.destroyWindow(self.window_name)
        
        large_polygon = self.upsize_polygon()
        # return large_polygon.astype("int32")
        return np.array([self.points]).astype("int32")

def annotate(batch):
    accession = input("tell me the accession ({})".format(batch))
    replicate = input("tell me the replicate ({})".format(batch))
    confirmation = input("accession: " + accession + " replicate: " + replicate + " confirm: y/n")
    if confirmation == "y" or confirmation == "":
        print("confirmed")
        return accession, replicate
    else:
        print("not confirmed, enter again")
        annotate(batch)


# ============================================================================

# if __name__ == "__main__":
#     pdrawer = PolygonDrawer("Polygon", )
#     image = pdrawer.run()
#     cv2.imwrite("polygon.png", image)
#     print("Polygon = %s" % pd.points)


# %% 3. Manually draw individual plant masks
# drawing = False	 # true if mouse is pressed 
# mode = True		 # if True, draw rectangle. 
# ix, iy = -1, -1

# # Load plant data

# # Create column for bbox coordinates

# # Iterate over batches
# for batch in batchname_lst:
#   # Open batch image
    
#   # Iterate over plants in batch
  
#   # Display accession and repetition; instruct to draw bounding box
  
#   # Store bounding box coords in dataframe
  
#   # save/overwrite dataframe on disk
    
# img = np.zeros((512, 512, 3), np.uint8) 
# cv2.namedWindow('image') 
# cv2.setMouseCallback('image', draw_circle) 

# while(1): 
# 	cv2.imshow('image', img) 
# 	k = cv2.waitKey(1) & 0xFF
# 	if k == ord('m'): 
# 		mode = not mode 
# 	elif k == 27: 
# 		break

# cv2.destroyAllWindows() 

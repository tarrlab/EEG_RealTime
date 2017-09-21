# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 18:40:08 2017

@author: Austin Marcus
"""

#import everything
from real_time_epoch_from_fieldtrip_biosemi_buffer import biosemi_fieldtrip_recent_epochs
import numpy as np
import matplotlib.pyplot as plt
import mne
#from mne.realtime import StimServer
#from mne.realtime import StimClient
from mne.realtime import FieldTripClient
import time
import random, datetime, time, glob, cv2
from psychopy import visual, core, event, data, gui
from ctypes import windll
import time


class ImageResponse:
    def __init__(self, im_to_show, code, window, imsize, pport_addr):
        self.im_to_show = im_to_show
        self.code = code
        self.window = window
        self.imsize = imsize
        self.pport_addr = pport_addr
        
        #set up fieldtrip client
        self.rt_client = FieldTripClient(host='localhost', port=1972,
        		             tmax=150, wait_max=10) 
        self.rt_client.__enter__()
        # FieldTrip buffer stuff
        self.recent_epochs_obj = biosemi_fieldtrip_recent_epochs(self.rt_client, n_recent_event = 1,
                                                        trial_tmax = 0.5, event_types = [code])                                              
        # where to send triggers
        #self.pport_addr = 0xcff8
        ImageResponse.pport = windll.inpoutx64
    
    def send_trigger(self, trig_val):
        pport.Out32(self.pport_addr, trig_val)   
    
    # reset parport bits all to 0
    def reset_trigger(self):
        pport.Out32(self.pport_addr, 0)
    
    # get recent epochs from FieldTrip buffer
    def get_recent(self):
        return self.recent_epochs_obj.get_recent_epochs()
       
    def get_image_response(self):
        self.reset_trigger()
        display = visual.ImageStim(win=self.window, image=self.im_to_show, units="pix", size=self.imsize) 
        display.draw()
        self.window.flip()
        self.send_trigger(self.code)
        self.reset_trigger()
        #display image for 500ms
        core.wait(0.5)
        recent_epochs, recent_event_list = self.get_recent()
        response_data = (recent_epochs, recent_event_list)
        return response_data


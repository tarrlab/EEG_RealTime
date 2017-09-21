# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 17:39:20 2016
Most recent working modification: Thur Oct 13 2016

@author: Ying Yang, yingyan1@andrew.cmu.edu
@author: Austin Marcus, aimarcus@andrew.cmu.edu
"""
import sys
paths = ['C:\\ExperimentData\\YingYang\\tools\\mne-python-master\\',
         'C:\\FieldTrip\\fieldtrip-20160810\\realtime\\src\\buffer\\python\\',
         'C:\\ExperimentData\\YingYang\\Real_time\\Initial_test_script\\']
for path0 in paths:
    sys.path.insert(0,path0)    
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


'''
class trigger():
    
    
    
    def __init__(self):
        self.address = 0xcff8
        self.pport = windll.inpoutx64
        self.events = [100,200]
        
    def send():
        pass        
        
        
    def reset():
        pass
        '''
# testing: object wrapper for entire session;
# foregoing internal socket structure for 
# just FieldTrip buffer client, presentation and
# trigger sends all in one module
# works, as of 10/13/2016!
"""
class session():
    
    def __init__(self, rt_client, pport_addr):
        
"""        


rt_client = FieldTripClient(host='localhost', port=1972,
		             tmax=150, wait_max=10) 
rt_client.__enter__()
# FieldTrip buffer stuff
recent_epochs_obj = biosemi_fieldtrip_recent_epochs(rt_client, n_recent_event = 1,
                                                event_types = [32, 64, 100, 128, 255])
                                                
#initialize data list
data_list = list()                                                

# where to send triggers
pport_addr = 0xcff8
pport = windll.inpoutx64


#get subject info from dialog box
dialog = gui.Dlg(title="EEG Realtime Localizer")
dialog.addText("Subject Info")
dialog.addField("Subject ID:")
dialog.addField("Age:")
dialog.addField("Handedness:")

subj_info = dialog.show()

#while not subj_info.OK:
#    subj_info = dialog.show()

subj_id = subj_info[0]
subj_age = subj_info[1]
subj_hand = subj_info[2]    
    
#start data file
datafile = open("Subject{}_EEGRealtimeLocalizer_log.txt".format(subj_id), "w+")
datafile.write("\t\t\t\t\t\t\tEEG Realtime Localizer: Experiment log")
datafile.write("\n")
datafile.write("Current time and date: {}\n".format(datetime.datetime.now()))
datafile.write("Subject info:\n")
datafile.write("ID: {}\tAge: {}\tHandedness: {}".format(subj_id, subj_age, subj_hand))
datafile.write("\n")
datafile.write("Block\t" \
               "Trial#\t" \
               "StimType\t" \
               "OneBack?\t" \
               "Correct?\t" \
               "RT\t" \
               "StimStart\t" \
               "StimDur\t\t" \
               "FixStart\t" \
               "FixDur\n")

# timers
timer1 = core.Clock()
timer2 = core.Clock()
'''
# stimulus display stuff
mywin = visual.Window([800, 600], monitor="testMonitor", units="deg")  # display window
right_cb = visual.RadialStim(mywin, tex='sqrXsqr', color=1, size=5,
                     visibleWedge=[0, 180], radialCycles=4,
                     angularCycles=8, interpolate=False,
                     autoLog=False)                                         # right checkerboard
left_cb = visual.RadialStim(mywin, tex='sqrXsqr', color=1, size=5,
                    visibleWedge=[180, 360], radialCycles=4,
                    angularCycles=8, interpolate=False,
                    autoLog=False)                                          # left checkerboard
fixation = visual.PatchStim(mywin, color=-1, colorSpace='rgb', tex=None,
                    mask='circle', size=0.2)                                # fixation

# events
ev_list = []
   '''     
# set parport bits to trig_val
def send_trigger(trig_val, pport):
    pport.Out32(pport_addr, trig_val)        
# reset parport bits all to 0
def reset_trigger(pport):
    pport.Out32(pport_addr, 0)

# get recent epochs from FieldTrip buffer
def get_recent(recent_epochs_obj):
    return recent_epochs_obj.get_recent_epochs()
        
   
   
sfreq = 100.0
   
#====================================   
'''
# start with fixation for 0.75 sec
fixation.draw()
mywin.flip()
timer1.reset()
timer1.add(0.75)
'''
#create a window
mywin = visual.Window([800, 800], monitor="testMonitor", units="deg")

first_ev = 100

#right_cb.draw()
#fixation.draw()
send_trigger(first_ev, pport)
mywin.flip()
timer1.reset()
reset_trigger(pport)
timer1.add(0.75)

while timer1.getTime() < 0:
    pass
        
#fixation.draw()
mywin.flip()
        
time.sleep(0.5)

########################
####load in stimuli#####
########################

#update all paths as necessary

#load faces
faces = []
face_ims = glob.glob("C:/ExperimentData/YingYang/Real_time/Initial_test_script/Stimuli/Faces/*.jpg")

for image in face_ims:
    faces.append(image)

#load objects
objects = []
object_ims = glob.glob("C:/ExperimentData/YingYang/Real_time/Initial_test_script/Stimuli/Objects/*.jpg")

for image in object_ims:
    objects.append(image)
    
#load scenes
scenes = []
scene_ims = glob.glob("C:/ExperimentData/YingYang/Real_time/Initial_test_script/Stimuli/Scenes/*.jpg")

for image in scene_ims:
    scenes.append(image)

#instruction text
intro_text = visual.TextStim(win=mywin, 
                             text="Press the space bar if an image repeats.",
                             wrapWidth=20,
                             alignHoriz='center',
                             alignVert='center')

intro_text.draw()
mywin.flip()

#wait for keypress
presses = event.waitKeys()

#begin experiment
begin_text = visual.TextStim(win=mywin, 
                             text="Press any key to begin.",
                             wrapWidth=20,
                             alignHoriz='center',
                             alignVert='center')

begin_text.draw()
mywin.flip()

#wait for keypress
presses = event.waitKeys()
    

#define the fixation display
fixation = visual.ShapeStim(win=mywin, vertices=((0,-8), (0,8), (0, 0), (8, 0), (-8, 0)),
                            lineWidth=2,
                            size=.05,
                            closeShape=False)

#define the break-signalling fixation
break_fix = visual.ShapeStim(win=mywin, vertices=((0,-8), (0,8), (0, 0), (8, 0), (-8, 0)),
                            lineWidth=2,
                            size=.05,
                            closeShape=False,
                            lineColor='black')
#program control
num_blocks = 1

#block loop
cur_block = 1
while cur_block <= num_blocks:
    
    #initialize helper variables for each block
    #max 90 faces, 45 objects, 45 scenes
    #total 200 stimuli shown per block, 20 repeated
    last_shown = ""     #save recent image for one-back condition
    faces_shown = 0     #track total faces shown
    objects_shown = 0   #track total objects shown
    scenes_shown = 0    #track total scenes shown
    total_shown = 0     #track total stims seen by participant
    stims_shown = 0     #track total distinct stims shown
    
    #shuffle everything
    random.shuffle(faces)
    random.shuffle(objects)
    random.shuffle(scenes)
    
    #get new random one-back matrix
    oneback_matrix = [0] * 200
    
    for i in range(0,200,10):
        oneback_matrix[i] = 1
    
    #first image shown can't be a repeat, so
    #shuffle until first index is 0, and 
    #until no two one-back conditions are within 
    #5 trials of each other
    checked = False
    while checked == False:
        if oneback_matrix[0] == 1:
            random.shuffle(oneback_matrix)
            continue
        tooclose = False
        for i in range(0,196):
            #check if one-back conditions within 5 trials
            if oneback_matrix[i] == 1 and \
               (oneback_matrix[i+4] == 1 or \
               oneback_matrix[i+3] == 1 or \
               oneback_matrix[i+2] == 1 or \
               oneback_matrix[i+1] == 1):
                   #reshuffle and start over
                   random.shuffle(oneback_matrix)
                   tooclose = True
                   break
        if tooclose == True:
            continue
        else:
            checked = True
    
    fixation.draw()
    mywin.flip()
    
    #get start time
    exp_start_time = time.time()
    
    core.wait(5.0)
    
    #core program loop
    while stims_shown < 180 and total_shown < 200:
    #while stims_shown < 4:
        #get recent events
        recent_epochs, recent_event_list = get_recent(recent_epochs_obj)
        recent_epochs.apply_baseline(baseline = (None,0))
        #recent_epochs.resample(sfreq = sfreq)
        data_list.append((recent_epochs, recent_event_list))
       
        #if we've reached a one-back condition, display the last 
        #image that was displayed
        if oneback_matrix[total_shown] == 1:
            to_display = last_shown
            #set trigger value
            trig_val = 255
            stim_type = "Repeat"
            oneback = "Yes"
        else:
            oneback = "No"
            rand = random.randint(1,3)
            if rand == 1:
                #first, check that face count hasn't maxed out
                if faces_shown >= 90:
                    continue
                #show a face
                stim_type = "Face"
                to_display = faces[faces_shown]
                faces_shown += 1
                #set trigger value
                trig_val = 32
            elif rand == 2:
                #first, check that object count hasn't maxed out
                if objects_shown >= 45:
                    continue
                #show an object
                stim_type = "Object"
                to_display = objects[objects_shown]
                objects_shown += 1
                #set trigger value
                trig_val = 64
            else:
                #first, check that scene count hasn't maxed out
                if scenes_shown >= 45:
                    continue
                #show a scene
                stim_type = "Scene"
                to_display = scenes[scenes_shown]
                scenes_shown += 1
                #set trigger value
                trig_val = 128
                
        #whichever image was chosen, display it
        display = visual.ImageStim(win=mywin, image=to_display, units="pix", size=250) 
        display.draw()
        #trigger the parport
        send_trigger(trig_val, pport)
        stim_start = time.time()
        stim_onset = time.time() - exp_start_time
        mywin.flip()
        #reset trigger value
        reset_trigger(pport)
        #display image for 500ms
        core.wait(0.5)
        
        #get timing information
        stim_end = time.time()
        stim_dur = stim_end - stim_start
        fix_start = time.time()
        fix_onset = time.time() - exp_start_time
        
        #draw fixation
        fixation.draw()
        mywin.flip()
        
        #get reaction time
        timeBefore = time.time()
        presses = event.waitKeys(1.0)
        timeAfter = time.time()
        trial_rt = timeAfter - timeBefore
        
        if(presses and presses[0] == "space" and oneback_matrix[total_shown] == 1):
            #correct response, handle 
            #print "Correct!"
            correct = "Y"
        elif(((presses and presses[0] != "space") or (not presses)) and oneback_matrix[total_shown] == 1):
            #incorrect response, handle
            #print "Incorrect response"
            correct = "N"
        else:
            correct = "n/a"
        #if not a one-back condition, save the most recently-displayed
        #image, and increment the total per-block distinct stim counter
        if oneback_matrix[total_shown] == 0:
            last_shown = to_display
            stims_shown += 1
        #increment total stim counter
        total_shown += 1
            
        #wait out the remaining time
        core.wait(1.0 - trial_rt)
        
        #get timing information
        end_fix = time.time()
        fix_dur = end_fix - fix_start
        
        #break every 25 trials
        if (total_shown + 1) % 25 == 0:
            break_fix.draw()
            mywin.flip()
            presses = event.waitKeys()
            fixation.draw()
            mywin.flip()
            core.wait(2.0)
            
        datafile.write("{}\t{}\t{}\t\t{}\t\t{}\t\t{}\t{}\t{}\t{}\t{}\n".format(cur_block, \
                                                                   total_shown, \
                                                                   stim_type, \
                                                                   oneback, \
                                                                   correct, \
                                                                   trial_rt, \
                                                                   stim_onset, \
                                                                   stim_dur, \
                                                                   fix_onset, \
                                                                   fix_dur))
        datafile.flush()                                                          
    
    wait_text = visual.TextStim(win=mywin, text="You have finished this block. Press any key to continue.")
    
    presses = event.waitKeys()
    if presses:
        cur_block += 1
        continue
    
    
    

data_array = data_list[0][0].get_data()
for l in range(1,len(data_list)):
    data_array = np.concatenate((data_array, data_list[l][0].get_data()), axis = 0)
    
trigger_array = np.array( [ data_list[l][1][0][1] for l in range(len(data_array)) ])

import scipy.io
mat_dict = dict(data_array = data_array, trigger_array = trigger_array, times = recent_epochs.times)
scipy.io.savemat('C:\ExperimentData\YingYang\Real_time\%s_localizer_data_20170228.mat' %subj_id, mat_dict)
    
    
ave_data = data_array.mean(axis = 0)    

#plt.imshow(ave_data[96:128,:])
plt.plot(recent_epochs.times, ave_data[96:128].T)


categories = ['face','object', 'scene']
cat_trigger_id = [32, 64, 128]
plt.figure()
for l in range(len(cat_trigger_id)):
    tmp_ave = data_array[trigger_array == cat_trigger_id[l]].mean(axis = 0)
    _ = plt.subplot(1,len(cat_trigger_id),l+1)
    _ = plt.plot(recent_epochs.times, tmp_ave[96:128].T)
    _ = plt.ylim([-40, 40])
    _ = plt.title(categories[l])
    



datafile.close()          
mywin.close()
#core.quit()

'''
for ii in range(0,40):
    # query the FieldTrip buffer
    recent_epochs, recent_event_list = get_recent(recent_epochs_obj)
    print recent_event_list
    print recent_epochs
    
    # determine stim to display
    if recent_event_list[-1][1] == 100.0:
        next_ev = 200
        left_cb.draw()
    else:
        next_ev = 100
        right_cb.draw()
    
    # draw the stimulus, send trigger and flip window
    fixation.draw()
    send_trigger(next_ev, pport)
    mywin.flip()
    timer1.reset()
    reset_trigger(pport)
    timer1.add(0.75)
    
    # display stim for 0.75 sec
    while timer1.getTime() < 0:
        pass
    
    #clear stim from screen
    fixation.draw()
    mywin.flip()
    
    # naive "data processing" during ISI to test timing
    sum = 0
    for i in range(len(recent_event_list)):
        sum += recent_event_list[i][0]
    print sum
    
    # ISI - allow time for trigger to reach buffer client
    # (total loop time ~1.1 sec)
    time.sleep(0.8)
    ''' 
    
"""
#== visualizing the average
ave = recent_epochs.average()
ave.plot()

avedata = ave.data
avedata_sumsq = (avedata**2).sum(axis = 1)

#==============================
import matplotlib.pyplot as plt
sort_ind = np.argsort(avedata_sumsq)

plt.figure()
plt.plot(avedata_sumsq[sort_ind])

print [ave.info['ch_names'][i] for i in sort_ind[96::]]

plt.figure()

data = recent_epochs.get_data()


recent_epochs.save('C:\ExperimentData\YingYang\Real_time\checkerboard_realtime_test_20170228-epo.fif.gz')

import scipy.io
mat_dict = dict(data = recent_epochs.get_data(),  avedata = avedata)
scipy.io.savemat('C:\ExperimentData\YingYang\Real_time\checkerboard_realtime_test_20170228_data_matrix.mat', mat_dict)
"""
        
        

'''
if recent_event_list[-1][1] == 100:
    self.left_cb.draw()
    self.send_trigger(200)
    self.mywin.flip()
    self.timer1.reset()
    self.reset_trigger()
    self.timer1.add(0.75)
    
    while self.timer1.getTime() < 0:
        pass
'''
#mywin.close()

    
        
'''        
# object wrapper for FieldTrip buffer client server session
class server():
    
    def __init__(self, session_port, rt_client, stim_server):
        print "Initializing server"
        self.session_port = session_port
        #self.rt_client = FieldTripClient(host='localhost', port=1972,
	  #	             tmax=150, wait_max=10)
        self.rt_client = rt_client
        self.recent_epochs_obj = biosemi_fieldtrip_recent_epochs(self.rt_client, n_recent_event = 1,
                                                        event_types = [100, 200])
        #self.server = StimServer(port=self.session_port)
        self.stim_server = stim_server
    
    # start running 'experiment' - control client
    def run(self):
        print "RUNNING SERVER"
        self.stim_server.start(verbose=True)
        #self.client_session = client(self.session_port, 0xcff8)
        #self.client_session.run()
        first_ev = 100
        #time.sleep(1)
        self.put_trigger(first_ev)
        
        time.sleep(15)        
        recent_epochs, recent_event_list = self.get_recent()
        print recent_event_list
        

        # check recent_event_list for trigger type to determine next stim - see if 
        # triggers are getting properly read in from FieldTrip buffer
        for ii in range(0,9):
            time.sleep(1)
            if recent_event_list[-1][1] == 100.0:
                next_ev = 200
            else:
                next_ev = 100
            self.put_trigger(next_ev)
            recent_epochs, recent_event_list = self.get_recent()
            print 'DEBUG~~contents of buffer list: {}'.format(recent_event_list[-1][1])
                
        recent_epochs.resample(512.0)
        recent_epochs.apply_baseline((None,0))

        # test to avoid socket timeout error
        time.sleep(1)
        
    # send a trigger to the presentation object
    def put_trigger(self, trig_val):
        self.stim_server.add_trigger(trig_val)
        
    # get recent epochs from FieldTrip buffer
    def get_recent(self):
        return self.recent_epochs_obj.get_recent_epochs()
        
    
# object wrapper for presentation client session
class client():
    
    def __init__(self, session_port, trig_addr):
        print "Initializing client"
        # IPC stuff
        self.session_port = session_port
        self.pport = windll.inpoutx64
        self.trig_addr = trig_addr
        self.stim_client = StimClient('localhost', port=self.session_port)
        
         # timers
        self.timer1 = core.Clock()
        self.timer2 = core.Clock()

    # run 'experiment'
    def run(self):
        print "RUNNING CLIENT"   
                
        # stimulus display stuff
        self.mywin = visual.Window([800, 600], monitor="testMonitor", units="deg")  # display window
        self.right_cb = visual.RadialStim(self.mywin, tex='sqrXsqr', color=1, size=5,
                             visibleWedge=[0, 180], radialCycles=4,
                             angularCycles=8, interpolate=False,
                             autoLog=False)                                         # right checkerboard
        self.left_cb = visual.RadialStim(self.mywin, tex='sqrXsqr', color=1, size=5,
                            visibleWedge=[180, 360], radialCycles=4,
                            angularCycles=8, interpolate=False,
                            autoLog=False)                                          # left checkerboard
        self.fixation = visual.PatchStim(self.mywin, color=-1, colorSpace='rgb', tex=None,
                            mask='circle', size=0.2)                                # fixation
        
        # events
        self.ev_list = []
        
        # start with fixation for 0.75 sec
        self.fixation.draw()
        self.mywin.flip()
        self.timer1.reset()
        self.timer1.add(0.75)
        
        trig = self.retrieve(0.2)
        print "DEBUGGING: trig = {}".format(trig)
        self.reset_trigger()
        if trig == 100:
            self.right_cb.draw()
        
        self.send_trigger(trig)
        self.mywin.flip()
        self.timer1.reset()
        self.reset_trigger()
        
        while self.timer1.getTime() < 0:
            pass
        
        self.mywin.close()
        

        # run 10 sample trials
        for ii in range(10):
            self.reset_trigger()
    
    
            # testing: tie trigger value to trigger sent from server session
            trig = self.retrieve(0.2)
            
         
            #if trig is not None:
            #    ev_list.append(trig)  # use the last trigger received
            #else:
            #    ev_list.append(ev_list[-1])  # use the last stimuli
         
            while trig is None:
                trig = self.retrieve(0.2)
              
            self.ev_list.append(trig)   
            print ev_list
         
            # draw left or right checkerboard according to ev_list
         
            # pyport.setData(255) # set parport pins all high   
            
            if self.ev_list[ii] == 200:
                self.left_cb.draw()
            else:
                self.right_cb.draw()
                
            self.reset_trigger()  # set parport pins all low
            # pyport.setData(0)   # set parport pins all low
         
            self.fixation.draw()  # draw fixation
            self.send_trigger(trig)    # set parport pins to latest trigger 
            self.mywin.flip()  # show the stimuli
         
            self.timer1.reset()  # reset timer
            self.timer1.add(0.75)  # display stimuli for 0.75 sec
         
            # return within 0.2 seconds (< 0.75 seconds) to ensure good timing
            #trig = stim_client.get_trigger(timeout=0.2)
            
            # testing trigger retrieval from server session
            #pport.Out32(0xcff8, trig)
         
            # wait till 0.75 sec elapses
            while self.timer1.getTime() < 0:
                pass
            
            self.fixation.draw()  # draw fixation
            self.mywin.flip()  # show fixation dot
         
            self.timer2.reset()  # reset timer
            self.timer2.add(0.25)  # display stimuli for 0.25 sec
         
            # display fixation cross for 0.25 seconds
            while self.timer2.getTime() < 0:
                pass
            
            time.sleep(1)
        self.mywin.close()  # close the window

    # get trigger from server session        
    def retrieve(self, time_val):
        return self.stim_client.get_trigger(timeout=time_val)
        
    # send trigger to BioSemi by setting parport bits
    def send_trigger(self, trig_val):
        self.pport.Out32(self.trig_addr, trig_val)
        
    # reset parport bits all low
    def reset_trigger(self):
        self.pport.Out32(self.trig_addr, 0)
'''     

#with FieldTripClient(host='localhost', port=1972,
#		             tmax=150, wait_max=10) as rt_client:  
#
#    expt_session = session(rt_client, 0xcff8)
#    expt_session.run()
'''
client_session = client(4218, 0xcff8)
print "ABOUT TO RUN CLIENT MAYBE"
client_session.run()



# debug

from psychopy import visual
import sys
paths = ['C:\\ExperimentData\\YingYang\\tools\\mne-python-master\\']
for path0 in paths:
    sys.path.insert(0,path0)
from mne.realtime import StimClient
from psychopy import core
from ctypes import windll
# create a port object to use in the session
pport = windll.inpoutx64

# ===debugging===
rt_client = FieldTripClient(info = None, host = "localhost", port = 1972, 
                            tmax = 150, wait_max = 10)
rt_client.__enter__()

pport.Out32(0xcff8, 0)  # set trigger pins to low
    
# testing: tie trigger value to trigger sent from server session
trig_seq = [100, 200, 100, 100, 200]    
    
for i in range(len(trig_seq)):
    trig = trig_seq[i]
    pport.Out32(0xcff8, 0)  # set parport pins all low
    time.sleep(1)
    pport.Out32(0xcff8, trig)    
    time.sleep(0.1)
    
    pport.Out32(0xcff8, 0)  # set parport pins all low
    time.sleep(1)
    
    
ftc = rt_client.ft_client
event_list = list(ftc.getEvents())
#for e in event_list[::-1]:
#    print e
    
H = ftc.getHeader()    
current_nSamples = H.nSamples
event_types = [100, 200]
for e in event_list[::-1]:
    # get event type and event sample
    str_e =(str(e)).replace(":","\n").replace('[',' ').replace(']',' ').split()
    # hard coded: split at n, type, trigger, value, [100], Sample, 470968, Offset, 0, Duration, 0
    current_event_ind = int(str_e[5])
    current_event_type = float(str_e[3]) # '[100]'
    if current_event_type in event_types:
        print (current_event_ind,current_event_type)

                

#recent_epochs_obj = biosemi_fieldtrip_recent_epochs(rt_client, n_recent_event = 1)
#recent_epochs, recent_event_list = recent_epochs_obj.get_recent_epochs()


#=========================
# number of trials in total
n_trial = 100                     
with FieldTripClient(host='localhost', port=1972,
		             tmax=150, wait_max=10) as rt_client:
    recent_epochs_obj = biosemi_fieldtrip_recent_epochs(rt_client, n_recent_event = 1,
                                                        event_types = [100, 200])
    
    
            
        # send a testing trigger
        stim_server.start(verbose=True)
        # Just some initially decided events to be simulated
        # Rest will decided on the fly
        #ev_list = np.tile(np.array([100,200]),[5,1]).ravel()
        #for ii in range(len(ev_list)):
            # Tell the stim_client about the next stimuli
            #time.sleep(1)
            #stim_server.add_trigger(ev_list[ii])
            # Collecting data
        
        first_ev = 100
        time.sleep(1)
        stim_server.add_trigger(first_ev)
        
        time.sleep(1)        
        recent_epochs, recent_event_list = recent_epochs_obj.get_recent_epochs()
        print recent_event_list
        
        # check recent_event_list for trigger type to determine next stim - see if 
        # triggers are getting properly read in from FieldTrip buffer
        for ii in range(0,9):
            time.sleep(1)
            if recent_event_list[-1][1] == 100.0:
                next_ev = 200
            else:
                next_ev = 100
            stim_server.add_trigger(next_ev)
            recent_epochs, recent_event_list = recent_epochs_obj.get_recent_epochs()
            print 'DEBUG~~contents of buffer list: {}'.format(recent_event_list[-1][1])
                
        recent_epochs.resample(512.0)
        recent_epochs.apply_baseline((None,0))
        
        # test to avoid socket timeout error
        time.sleep(1)
    
  '''  
   
'''
Created on Apr 4, 2012

@author: lanquarden
'''

import msgParser
import carState
import carControl
import learner

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        
        self.state = carState.CarState()
        
        self.control = carControl.CarControl()
        
        # the max angle in rad that the car can turn left or right
        self.steer_lock = 0.785398

        self.max_speed = 100
        self.prev_rpm = None

        self.rl = learner.DriverLearner()
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    # Changes the effectors - acts on the actions computed by the rl algo
    def drive(self, msg):
        self.state.setFromMsg(msg)       
        print "dist raced", self.state.distRaced 

        RL_DRIVE = True
        if RL_DRIVE:
            steering, accel, reset = self.rl.learnAndGetNextAction(self.state)

            self.control.setSteer(steering)
            self.control.setAccel(accel)

            if reset:
                self.control.setMeta(1)
        else:
            self.steer()
            self.speed()
        
        self.gear()
            
        return self.control.toMsg()
    
    # [-1, 1], means full left to full right, corresponds to steer_lock
    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
    # Not touching this for now
    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        
        if self.prev_rpm == None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False

        if up and rpm > 7000:
            gear += 1
            gear = min(gear, 5)
        
        if not up and rpm < 3000:
            gear -= 1
            gear = max(gear, 1)
        
        self.prev_rpm = rpm
        self.control.setGear(gear)
    
    # [0, 1] - no gas to full gas
    # eventually need to change to incorporate brakes
    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        
        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        
        self.control.setAccel(accel)

    def onShutDown(self):
        self.rl.logWeights()
        self.rl.logRewards()
        self.rl.cleanup()
    
    def onRestart(self):
        self.rl.logWeights()
        self.rl.logRewards()   
        self.control.setMeta(0)
        
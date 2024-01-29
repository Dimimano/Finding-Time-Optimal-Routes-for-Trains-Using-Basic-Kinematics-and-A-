#LIBRARIES
import math
import argparse
import ast

#GLOBAL
TRAIN_LENGTH_LIMIT = 4000
MAX_ACC_LIMIT = 5
MIN_ACC_LIMIT = 5
MIN_DIFFERENCE = .000000001

def parseVariables(filename):
    '''
    Parses the input data from the text file, that is given as input during the program's execution.
    '''
    variables = {}

    with open(filename, 'r') as file: # Read the text file.
        for line in file.readlines():
            if '=' in line:
                var_name, var_value = line.split('=', 1)
                var_name = var_name.strip()

                try:
                    # Safely evaluate the right-hand side as a Python literal.
                    var_value = ast.literal_eval(var_value.strip())
                except (ValueError, SyntaxError) as e:
                    # Handle the case where the value is not a valid Python literal.
                    print(f"Error parsing variable {var_name}: {e}")
                    continue

                variables[var_name] = var_value

    return variables

def positionsToSegments(positions):
    '''Given a list of positions, it converts it into a list of segment lengths.'''
    return [positions[0]] + [positions[i+1] - positions[i] for i in range(len(positions) - 1)]

def storeVariables(variables):
    '''
    Convert the input data stored into the variables dictionary into Python variables.
    '''
    # init state
    init = variables['init']
    initV = init[1]
    initPos = init[0]

    # goal state
    goal = variables['goal']
    goalV = goal[1]
    goalPos = goal[0]

    # Train characteristics
    trainLength = variables['trainLength']
    Acc = variables['maxAcc']
    Dec = variables['maxDec']
    trainMaxSpeed = variables['trainMaxSpeed']
    trainChar = (trainLength,Acc,Dec,trainMaxSpeed)

    if 'maxSpeeds' in variables:
        # Path characteristics
        maxSpeeds = variables['maxSpeeds']
        positions = variables['positions']
        lengths = positionsToSegments(positions)
        numberOfSegments = len(positions)
        segmentIndices = [integer for integer in range(1,numberOfSegments+1)]
        pathChar = (maxSpeeds,positions,lengths,segmentIndices)
    else:
        pathChar = None
    
    return initV, initPos, init, goalV, goalPos, goal, trainChar, pathChar

def speed(state):
    '''Returns the train's speed at a given state.'''
    
    return state[1]

def pos(state):
    '''Returns the train's position at a given state.'''
    
    return state[0]

def distance(state, positions):
    '''Returns the total distance traversed from the train, at a given state.'''
    
    hSgm, hExit = max(pos(state)[0]), pos(state)[1]
    return positions[hSgm-1] - hExit
    
def validateData(pathChar, trainChar):
    '''Checks the validity of the given data (train & path characteristics).'''
    
    #Extract train characteristics.
    trainLength, Acc, Dec, trainMaxSpeed = trainChar
    
    #Extract path characteristics.
    maxSpeeds, positions, lengths, segmentIndices = pathChar

    #Train length values must be positive and less than the length limit.
    if trainLength < 0 or trainLength > TRAIN_LENGTH_LIMIT: return False
    #Acceleration and deceleration values must be positive.
    if Acc < 0 or Dec < 0: return False
    #Acceleration and deceleration values must not exceed the correspoding limits.
    if Acc > MAX_ACC_LIMIT or Dec > MIN_ACC_LIMIT: return False
    
    print('Train characteristics are valid.')
    
    #Speed limit must be positive.
    if any(speed < 0 for speed in maxSpeeds): return False
    #Distances and lengths must be positive.
    if any(pos <= 0 for pos in positions) or any(length <= 0 for length in lengths): return False
    #First entries must be equal.
    if positions[0] != lengths[0]: return False
    #The difference of two consecutive positions elements must be equal to the correspoding lengths element.
    if any(positions[i]-positions[i-1] != lengths[i] for i in range(1,len(positions))): return False
    
    print('Path characteristics are valid.')
    
    return True

def validateStates(hSgm, hExit, tExit, trainChar, pathChar):
    '''Checks the validity of the states.'''
    
    #Extract train characteristics.
    trainLength, Acc, Dec, trainMaxSpeed = trainChar
    
    #Extract path characteristics.
    maxSpeeds, positions, lengths, segmentIndices = pathChar
    
    #Extract speed and position at init and goal.
    initV, initPos, goalV, goalPos, hExitGoal = speed(init), pos(init), speed(goal), pos(goal), goal[0][1]
    
    #Speed values must be positive.
    if initV < 0 or goalV < 0: return False
    #Speed values must not exceed the train's maximum speed.
    if initV > trainMaxSpeed or goalV > trainMaxSpeed: return False
    #Distance values must be positive.
    if initPos[1] < 0 or goalPos[1] < 0: return False
    #Indices must be positive or zero.
    if not all(index >= 0 for index in initPos[0]) or not (index >= 0 for index in goalPos[0]): return False
    
    #hExit cannot be larger than segment's hSgm length.
    if hExit > positions[hSgm-1]: return False
    
    #hExit at the goal state must not be larger than the last segment's length.
    hSgmtGoal = max(goalPos[0])
    hSegmentLength =  positions[hSgmtGoal-1]-positions[hSgmtGoal-2]
    if hExitGoal > hSegmentLength: return False

    #tExit at the goal state must not be larger than the tSgm's length.
    tSgmtGoal = min(goalPos[0])
    tExitGoal, tSegmentLength = positions[tSgmtGoal-1] - (positions[hSgmtGoal-1] - hExitGoal - trainLength), lengths[-1]
    if tExitGoal > tSegmentLength: return False

    #hExit and tExit in the init and goal states must have positive values.
    if hExit < 0 or tExit < 0 or hExitGoal < 0 or tExitGoal < 0 : return False

    print('Init and Goal states are valid.')
    
    return True 

def getSpeedLimit(trainMaxSpeed, maxSpeeds, tSgm, hSgm):
    '''Takes as input the train characteristics and the segments it is currently in, in order to calculate the speed limit.'''
    
    lowestVmax = min(maxSpeeds[tSgm-1:hSgm])
    
    return min(trainMaxSpeed,lowestVmax)

def MaxRoot(a,b,c):
    '''Returns the positive root.'''
    
    Delta = math.pow(b,2) - 4*a*c #discriminant
    t = (-b + math.sqrt(Delta))/(2*a) #positive root
    
    return t
    
def getMinT(currentAcc, currentV, tExit, hExit, speedLimit):
    '''Detects which type of event is upcoming, by computing the time tMin = min(t1, t2, t3) needed to reach each one of them.'''

    #Time needed for the tail to exit its segment.
    if currentAcc>0: t1 = MaxRoot((1/2)*currentAcc, currentV, -tExit)
    else: t1 = tExit/currentV 
    #Time needed for the head to enter a new segment.
    if currentAcc>0: t2 = MaxRoot((1/2)*currentAcc, currentV, -hExit)
    else: t2 = hExit/currentV
    #Time needed for the train to reach the speedLimit, provided that it is in accelerating mode.
    if currentAcc > 0 and speedLimit > currentV: t3 = (speedLimit - currentV)/currentAcc
    else: t3 = float("inf")

    return min(t1,t2,t3), t1, t2, t3

def initVariables(init, goal, trainChar, pathChar):
    '''Initialize all the necessary variables'''
    
    #Extract speed and position init and goal.
    initV, initPos, goalPos = init[1], init[0][0], goal[0][0]
    
    #Extract train characteristics.
    trainLength, Acc, Dec, trainMaxSpeed = trainChar
    
    #Extract path characteristics.
    maxSpeeds, positions, lengths, segmentIndices = pathChar
    
    #Tracks the train's speed.
    currentV = initV
    
    #Keeps the train's acceleration.
    currentAcc = Acc

    #hSgm and tSgm represent the indices of the segments in which head(T) and tail(T) reside respectively.
    tSgm, hSgm = min(initPos), max(initPos)
    
    #hExit and tExit concern the distance of head(T) and tail(T) from the end of segments hSgm and tSgm respectively.
    hExit = init[0][1]
    tExit = positions[tSgm-1] - (positions[hSgm-1] - hExit - trainLength)
    #If hExit = 0, hSmg is incremented by one and hExit is set to be equal to the length of next segment and init state is updated.                                           
    if hExit == 0:
        hSgm += 1
        hExit = lengths[hSgm-1]
        init = (([index for index in range(tSgm,hSgm+1)], hExit),initV)
        
    #Validate tExit and hExit in the init and goal states.
    #flag = validateStates(goal[0][1], hSgm, hExit, tExit, trainChar, pathChar)
    flag = True

    #The init state is the first entry stored in SP.
    SP = [init]
    
    #Keeps track of the time during the train's motion.
    time = [0]
    
    #Calculate speedLimit for the 1st iteration and update acceleration if speedLimit equal to currentV.
    speedLimit = getSpeedLimit(trainMaxSpeed, maxSpeeds, tSgm, hSgm)
    if speedLimit == currentV: currentAcc = 0

    #Calculate tMin for the 1st iteration.
    tMin, t1, t2, t3 = getMinT(currentAcc, currentV, tExit, hExit, speedLimit)
    
    return currentV, currentAcc, time, tSgm, hSgm, hExit, tExit, SP, flag, init, speedLimit, tMin, t1, t2, t3

def findNewState(trainChar, pathChar, xDec, hExit, v1Hash):
    '''The deceleration point calculated from the decelerate() function can be anywhere on the path. This function returns the train's state at that position.'''
    
    #Extract train characteristics.
    trainLength, Acc, Dec, trainMaxSpeed = trainChar
    
    #Extract path characteristics.
    maxSpeeds, positions, lengths, segmentIndices = pathChar

    for i in range(len(positions)): #Iterate on the positions, in order to find in which segment the position xDec lies.
        
        if xDec <= positions[i]: #xDec is in the segment between positions[i-1] and positions[i].
            hExit = positions[i] - xDec #Update hExit
            hSgm = i+1 #Update hSgm
            
            if lengths[i] - hExit >= trainLength: tSgm = i+1 #Covers the case where both head(T) and tail(T) are in the same segment.
            else: #If the tail is in a different segment, subtract the part of the train that is in each segment from the train's length, in order to find tSgm.
                
                total = lengths[i] - hExit #part of the train in segment i.
                
                for j in range(i-1,-1,-1): #Iterate on previous segments.
                    if lengths[j] - total >= trainLength: 
                        tSgm = j+1 #If the remaining part of the train is smaller that the length of segment j update tSgm.
                        break #and stop.
                    else: total += lengths[j] #else update total and continue searching.              
                    
            return (([index for index in range(tSgm,hSgm+1)], hExit),v1Hash) #State at xDec.
        
def addTwoStates(i, SP, time, v1Hash, currentAcc, trainChar, pathChar, xDec, hExit, v0, v1, Dec, d1):
    '''The decelerate() function might need to add two new states, one before and one after the train's deceleration. This function creates and returns these states.'''
                
    SP, time = SP[0:i+1], time[0:i+1] #SP and time are updated to include only entries until the state i.
                
    #Two new states are going to be added.
                
    #One after the train's acceleration or steady speed.
    if currentAcc != 0: t1Hash = (v1Hash - v1)/currentAcc #Duration of acceleration.
    else: t1Hash = (xDec - d1)/v1 #Duration of steady speed.
    time.append(time[-1] + t1Hash) #Update time.
    SP.append(findNewState(trainChar, pathChar, xDec, hExit, v1Hash)) #State at xDec.
                
    #One after the train's deceleration.
    t2Hash = (v1Hash - v0)/Dec #Duration of deceleration.
    time.append(time[-1] + t2Hash) #Update time.
    #The state at the start of the new segment will be added by ALGORITHM1.
                
    currentV, currentAcc = v0, 0 #Update speed and mode.
                
    #A solution was found.
    decFlag = True
    
    return i, SP, time, v1Hash, trainChar, pathChar, xDec, hExit, v0, Dec, currentV, currentAcc, decFlag, t1Hash, t2Hash

def addOneState(i, SP, time, v1Hash, v0, Dec):
    '''The decelerate() function might need to one new state, one after the train's deceleration (in case the train must decelerate immediately). This function creates and returns these states.'''
    
    SP, time= SP[0:i+1], time[0:i+1] #SP and time are updated to include only entries until the state i.
                    
    #Only one state is going to be added.
                    
    #After the train's deceleration.
    tHash = (v1Hash - v0)/Dec #Duration of deceleration.
    time.append(time[-1] + tHash) #Update time.
    #The state at the start of the new segment will be added by ALGORITHM1.
                    
    currentV, currentAcc = v0, 0 #Update speed and mode.
                    
    #A solution was found.
    decFlag = True
    
    return i, SP, time, v1Hash, v0, Dec, currentV, currentAcc, decFlag, tHash

def decelerate(trainChar, pathChar, SP, time, currentV, currentAcc, speedLimit, debug):
    '''ALGORITHM2: Backtracks in previous states stored in SP, searching for the optimal point to decelerate.
    The most complicated case occurs when head(T) is about to enter a new segment (t2 = tmin) with currentV higher than the new
    segment's speed limit. In this case the speed profile must be recomputed by adding a deceleration point at the latest possible time,
    when the train should start decelerating in order to reach the new segment with a speed equal to speedLimit.'''
    
    if debug: print(f'Train reached a segment with speed limit {speedLimit} with a speed equal to {currentV}.')
    
    #Extract train characteristics.
    trainLength, Acc, Dec, trainMaxSpeed = trainChar
    
    #Extract path characteristics.
    maxSpeeds, positions, lengths, segmentIndices = pathChar
    
    #If ALGORITHM2 can not find a deceleration point, the problem has no solution.
    decFlag = False
    
    #Total distance at the start of the new segment (last state).
    d = distance(SP[-1],positions)
        
    for i in range(len(SP)-2,-1,-1): #Iteratively backtrack to previous states (starting from the second to last).
        
        #The algorithm searches for previous parts of the speed profile (SP) during which the train was either accelerating
        #or maintaining its speed, starting from the most recent on.
        
        #v0 is the max speed of the new segment (speed-limit).
        #v1 is the speed of the train at the start of the SP part.
        #v2 is the speed of the train at the end of the SP part.
        v0, v1, v2 = speedLimit, speed(SP[i]), speed(SP[i+1])
        
        #Extract hExit for this part.
        hExit = pos(SP[i])[1]
        
        if v2 >= v1: #We are interested only in SP parts with a mode of acceleration or steady speed.
            
            if debug: print(f'i: {i}, v0: {v0}, v1: {v1}, v2: {v2}, hExit: {hExit}')
            
            #Determine the train's mode during the SP part.
            if v2==v1: currentAcc = 0
            else: currentAcc = Acc
            
            #Total distance at the start of the SP part.
            d1 = distance(SP[i],positions)
            #Calculate the optimal deceleration point between d1 and d.
            xDec = (math.pow(v0,2) - math.pow(v1,2) + 2*(currentAcc*d1 + Dec*d))/(2*(currentAcc + Dec))
            #Calculate the speed at that point.
            v1Hash = math.sqrt(math.pow(v1,2) + 2*currentAcc*(xDec - d1))
            
            if debug: print(f'd: {d}, d1: {d1}, xDec: {xDec}, v1Hash: {v1Hash}')
            
            if v1Hash > v1: #The train can accelerate until xDec.
                
                #Add two states. One before and one after the deceleration.
                i, SP, time, v1Hash, trainChar, pathChar, xDec, hExit, v0, Dec, currentV, currentAcc, decFlag, t1Hash, t2Hash = addTwoStates(i, SP, time, v1Hash, currentAcc, trainChar, pathChar, xDec, hExit, v0, v1, Dec, d1)
                
                if debug: print(f'The train can accelerate until xDec. New state: {SP[-1]}. Acc time: {t1Hash}, Dec time: {t2Hash}.')
                
                return decFlag, SP, time, currentV, currentAcc
                
            if v1Hash == v1: #The train must have steady speed until xDec (or decelerate immediately).
                    
                if xDec > d1: #The train must have steady speed until xDec.
                    
                    #Add two states. One before and one after the deceleration.
                    i, SP, time, v1Hash, trainChar, pathChar, xDec, hExit, v0, Dec, currentV, currentAcc, decFlag, t1Hash, t2Hash = addTwoStates(i, SP, time, v1Hash, currentAcc, trainChar, pathChar, xDec, hExit, v0, v1, Dec, d1)
                    
                    if debug: print(f'The train must have steady speed until xDec. New state: {SP[-1]}. Steady speed time: {t1Hash}, Dec time: {t2Hash}.')
                
                    return decFlag, SP, time, currentV, currentAcc
                    
                elif xDec == d1: #The train must decelerate immediately.
                    
                    #Add one state. One after the deceleration.
                    i, SP, time, v1Hash, v0, Dec, currentV, currentAcc, decFlag, tHash = addOneState(i, SP, time, v1Hash, v0, Dec)
                    
                    if debug: print(f'The train must decelerate immediately. Dec time: {tHash}.')
                    
                    return decFlag, SP, time, currentV, currentAcc
                
    return decFlag, SP, time, currentV, currentAcc

def equals(value1, value2):
    '''Checks whether two values are equal, allowing a minimum difference (due to the automatic rounding of float numbers with many digits).'''
    
    if abs(value1 - value2) <= MIN_DIFFERENCE: return True
    else: return False
    
def finalSegment(pathChar, trainChar, tSgm, hSgm, t2Flag, t1Flag, speedLimit):
    '''Updates hExit and hSgm when it is detected by ALGORITHM1, that the train is about to finish the final segment.
    This case happens mainly when t2=tMin, but there is a chance for t1=t2=tMin when l(final_segment) = trainLength.
    Hence, t2flag and t1flag are used to determine the specific case.'''
    
    #Extract train characteristics.
    trainLength, Acc, Dec, trainMaxSpeed = trainChar
    
    #Extract path characteristics.
    maxSpeeds, positions, lengths, segmentIndices = pathChar
    
    
    if t2Flag: #t2=tMin
        
        hExit = 0 #The train reached its destination.
        speedLimit = getSpeedLimit(trainMaxSpeed, maxSpeeds, tSgm, hSgm) #Update speed limit (using goalV -> last entry in maxSpeeds).
        hSgm -= 1 #There is no next segment. Revert the update of hSgm by ALGORITHM1
        t2Flag = False
        
        return  hExit, hSgm, t2Flag, t1Flag, speedLimit
        
    if t1Flag: #t1=t2=tMin
        
        tExit = trainLength #The train reached its destination.
        tSgm -= 1 #There is no next segment. Revert the update of tSgm by ALGORITHM1
        t1Flag = False
        
        return  tExit, tSgm, t2Flag, t1Flag  
    
def updatePathChar(goal, pathChar):  
    '''Updates the path characteristics in accordance to the goal state.
    In goal the train must reach the end of the last segment (if this is not the case, the last segment splits in two ones). 
    An additional segment is assumed, after the last one, with a speed limit equal to the desired speed of the train in goal.'''
    
    #Extract path characteristics.
    maxSpeeds, positions, lengths, segmentIndices = pathChar
    
    #Extract goal state.
    goalPos, goalV = goal 

    if set(goalPos[0]).issubset(set(segmentIndices)): #If the path characteristics contain the goal.
        
        finalSegment = (goalPos[0])[-1] #Set last segment of the goal to be the final segment.
        finalSegmentIndex = segmentIndices.index(finalSegment) #Find its index.
        
        #Update path characteristics in order to include only entries until the final segment.
        maxSpeeds, positions, lengths, segmentIndices = maxSpeeds[0:finalSegmentIndex+1], positions[0:finalSegmentIndex+1], lengths[0:finalSegmentIndex+1], segmentIndices[0:finalSegmentIndex+1]
        #Add a final max speed limit equal to goalV.
        maxSpeeds.append(goalV) 
    
    if goalPos[1] != 0: #If the goal is not at the end of a segment.
             
        #Split the final segment at goal and delete the second part.     
        positions[-1], lengths[-1] = positions[-1] - goalPos[1], lengths[-1] - goalPos[1]
        #Update goal position.
        goalPos = (goalPos[0],0)
    
    #Update goal state.
    goal = goalPos, goalV 
    
    pathChar = maxSpeeds, positions, lengths, segmentIndices
        
    return goal, pathChar

def equalStates(state1, state2):
    '''Checks whether two states are equal, allowing a minimum difference (in hExit and speed).'''
    
    #Extract position and speed of each state.
    state1Pos, state1V, state2Pos, state2V = state1[0], state1[1], state2[0], state2[1]
    
    if state1Pos[0] != state2Pos[0]: return False #Check if position (list of segments) is different.
    if abs(state1V-state2V) <= MIN_DIFFERENCE and abs(state1Pos[1]-state2Pos[1]) <= MIN_DIFFERENCE: return True #Check hExit and speed of each state.
    
    return False
    
def debugger(iterationCounter, speedLimit, currentAcc, t1, t2, t3, SP, time, p , currentV, tSgm, hSgm, hExit, tExit):
    '''Prints information for debugging purposes.'''
    
    print('---------------------------------------------------------------------')
    print(f'iterationCounter: {iterationCounter}')
    print(f'speedLimit: {speedLimit}, currentAcc: {currentAcc}, t1, t2, t3: {t1, t2, t3}')
    print(f'current State: {SP[-1]}')
    print(f'current time: {time[-1]}')
    print(f'hExit: {hExit}, tExit: {tExit}')
    print(f'Train moved {p} meters in {time[-1] - time[-2]} seconds, CurrentV: {currentV}, currentAcc: {currentAcc}')
    newState = (([index for index in range(tSgm,hSgm+1)],hExit),currentV)
    print(f'New state: {newState}')
    
def computeOptimalSP(goal, pathChar, trainChar, variables, debug, intermediate):
    '''ALGORITHM1: Computation of the optimal speed profile between an initial state init and a goal state goal, assuming that 
    there is a single path that can be used to reach the goal state, with no junctions across it.
    Iteratively detects events during the train's journey.'''
    
    #Extract train characteristics.
    trainLength, Acc, Dec, trainMaxSpeed = trainChar
    
    #Extract path characteristics.
    maxSpeeds, positions, lengths, segmentIndices = pathChar
    
    #Extract variables.
    currentV, currentAcc, time, tSgm, hSgm, hExit, tExit, SP, flag, init, speedLimit, tMin, t1, t2, t3 = variables
    goalV = speed(goal)
    
    iterationCounter = 0
    
    while(not equalStates(SP[-1],goal)): #while current state != goal
        
        iterationCounter += 1
        
        p = currentV*tMin + (1/2)*currentAcc*math.pow(tMin,2) #Update position.
        currentV = currentV + tMin*currentAcc #Update speed.
        tExit, hExit = tExit - p, hExit - p #Update tExit, hExit.
        time.append(time[-1] + tMin) #Update total time.

        if debug: debugger(iterationCounter, speedLimit, currentAcc, t1, t2, t3, SP, time, p , currentV, tSgm, hSgm, hExit, tExit) #Print info

        #The train's head is about to enter a new segment. Deceleration might be needed. 
        if equals(t2, tMin): 
            
            hSgm += 1 #The train's head is in the new segment.
            
            try: 
                hExit = lengths[hSgm-1] #Set hExit equal to the length of the new segment.
                speedLimit = getSpeedLimit(trainMaxSpeed, maxSpeeds, tSgm, hSgm) #Calculate the new speed limit.

            #This covers the case when the train is about to finish the last segment.
            except:
                t2Flag, t1Flag = True, False
                hExit, hSgm, t2Flag, t1Flag, speedLimit = finalSegment(pathChar, trainChar, tSgm, hSgm, t2Flag, t1Flag, speedLimit)

            if currentV > speedLimit: #The train will need to decelerate.
                
                #Store the achieved state before backtracking.
                SP.append((([index for index in range(tSgm,hSgm+1)],hExit),currentV))

                #Apply the algorithm that handles deceleration.
                decFlag, SP, time, currentV, currentAcc = decelerate(trainChar, pathChar, SP, time, currentV, currentAcc, speedLimit, debug)
                
                if not decFlag:
                    print('\nNo deceleration point could be found. There is no solution!')
                    return None, None
            
        #The train's tail is about to exit a previous segment. Acceleration might be needed. 
        if equals(t1, tMin):
            
            tSgm += 1 #The train's tail exited the previous segment.
            
            try:
                tExit = lengths[tSgm-1] #Set hExit equal to the length of the next segment.
                speedLimit = getSpeedLimit(trainMaxSpeed, maxSpeeds, tSgm, hSgm) #Calculate the new speed limit.
            #This covers the case when the train is about to finish the last segment.
            except:
                t2Flag, t1Flag = False, True
                tExit, tSgm, t2Flag, t1Flag = finalSegment(pathChar, trainChar, tSgm, hSgm, t2Flag, t1Flag, speedLimit)
                      
            if currentV < speedLimit: currentAcc = Acc #Acceleration
        
        #The train reached the speed limit. A steady speed must be maintained.
        if equals(t3, tMin) and currentV == speedLimit: currentAcc = 0
        
        SP.append((([index for index in range(tSgm,hSgm+1)],hExit),currentV)) #Update SP
        

        #At this point it is checked whether the train reached the goal state.
        #Note that in the case of a network, the algorithm might work on smaller/intermediate paths that do not include the goal state.
        #In that case, the train does not need to have a speed equal to goalV when reaching the end of the path.
        if not intermediate:
            #If the train reaches the goal position with a lower speed than goalV, then there is no solution.
            if pos(SP[-1]) == pos(goal) and abs(distance(SP[-1],positions)-distance(goal,positions))<=MIN_DIFFERENCE and currentV != goalV:
                print('\nIt is impossible to accelerate to goalV!')
                return None, None
            #If the path is an intermediate one, the train does not need to have a speed equal to goalV.
        else:
            if pos(SP[-1]) == pos(goal) and abs(distance(SP[-1],positions)-distance(goal,positions))<=MIN_DIFFERENCE:
                break
        
        if speedLimit == currentV: currentAcc = 0 #Update train's mode.
        
        if currentV != 0: tMin, t1, t2, t3 = getMinT(currentAcc, currentV, tExit, hExit, speedLimit) #Calculate tMin.
    
    return SP, time

if __name__ == "__main__":

    # Initialize the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--debug', nargs='?', type=bool, const=True, default=False, help='Enable debug mode.')

    # Parse the arguments.
    args = parser.parse_args()

    # Read the variables from the file.
    variables = parseVariables(args.filename)
    
    # Store the parsed variables as Python variables.
    initV, initPos, init, goalV, goalPos, goal, trainChar, pathChar = storeVariables(variables)

    flag = validateData(pathChar, trainChar) # Train & Path characteristics Validation
    if not flag: print('Not Valid Data')

    currentV, currentAcc, time, tSgm, hSgm, hExit, tExit, SP, flag, init, speedLimit, tMin, t1, t2, t3 = initVariables(init, goal, trainChar, pathChar) # Initialization

    variables = (currentV, currentAcc, time, tSgm, hSgm, hExit, tExit, SP, flag, init, speedLimit, tMin, t1, t2, t3)
    flag = validateStates(hSgm, hExit, tExit, trainChar, pathChar) # Init & Goal states Validation
    if not flag: print('Not Valid States')

    goal, pathChar = updatePathChar(goal, pathChar) # Update goal.
    SP, time = computeOptimalSP(goal, pathChar, trainChar, variables, args.debug, False) # Compute speed profile.
    
    if SP: #If a solution was found.
        print('\nTotal time: ', time[-1])

        print('\nOptimal Speed profile of the path\n')
        for i, (state, timeValue) in enumerate(zip(SP,time), start=1):
            print(f"State {i}: {state}, Time: {timeValue}")

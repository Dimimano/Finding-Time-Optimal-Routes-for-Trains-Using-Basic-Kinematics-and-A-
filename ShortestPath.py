from OptimalPathSpeedProfile import *
import networkx as nx
import copy
import argparse
import numpy as np

def extractStartDestination(G, initPos, goalPos):
    '''Extract the indices of the starting and the final segments, as well as the the starting and final network nodes.
     Note that in cases where the train is in multiple segments, the last segments of the initPos and goalPos are extracted.
     
    :param G: A networkx Graph in which the problem is defined on
    :param initPos: Tuple, the position of the init state (tuple -> list of segment indices + distance from end of segment).
    :param goalPos: Tuple, the position of the goal state.
    :return: openDict: Dictionary (key -> list of characters/nodes, value -> float) that contains the f values of all the discovered paths.
    :return: destinationNode: Char, the name of the final node (at the end of final segment).
    :return: start: List, sequence of nodes the define the starting path (given in init).
     '''

    #Indices of the starting and the destination segments.
    startSegment = initPos[0][-1] - 1 
    destinationSegment = goalPos[0][-1] - 1 

    #Extract start path and the destination node.
    for u,v,e in G.edges(data=True):
        if e['label'] == startSegment: start = (u,v) #Store the nodes in each side of the startSegment as a tuple.
        if e['label'] == destinationSegment: destinationNode = v #Store the final node.

    #Initialize openDict: contains all the discovered paths and is initialized to include only the start path, that is a sequence of nodes 
    # (u0, u1, ..., uk , sk ), defined by the segments (s0, .., sk ) in init.
    openDict = {}
    openDict[start] = 0 # the keys identify all the discovered paths, while the values concern the f values for a particular path.

    return openDict, destinationNode, start

def generatePaths(G, path):
    '''generatePaths() iterates through the outgoing edges (of the last vertex of the path), generating new paths
    (which are extensions of the current path, with a single or more segments added to it) that are stored temporarily in neighbors, before entering openDict.
    
    :param G: A networkx Graph in which the problem is defined on.
    :param path: List, contains the nodes of the path.
    :return: neighbors: List, containts the nodes of each path that was generated through the outgoing edges.'''
    
    neighbors = []
    node = path[-1] #last vertex of the path

    for u,v,e in G.edges(data=True):
        if u == node: neighbors.append(path + (v,)) #Iterate through the outgoing edges and generate paths

    return neighbors

def updateOpenDict(newPaths, dict, oldPath, timeDict=0):
    '''Updates the temporary versions of openDict (neighbors or successors) with the paths generated from function generatePaths() (stored in newPaths),
    replacing the previous not-extended entry (oldPath).
    
    :param newPaths: List, contains the paths generated from generatePaths().
    :param dict: Dictionary (neighbors or successors) (key -> list of characters/nodes, value -> float), contains the path of openDict which will be expanded.
    :param oldPath: List, containts the nodes of the path before its expansion.
    :param timeDict: Dictionary (default value = 0) (key -> list of characters/nodes, value -> list of floats), contains the time needed to reach a state in the speed profile of each path.
    :return: dict: Dictionary(neighbors or successors) (key -> list of characters/nodes, value -> float), contains the expanded versions of the generated paths in newPaths.'''

    if timeDict == 0: gValue = dict[oldPath] #If timeDict is not yet initialized, the algorithm is in the first iteration (no g values have been computed yet),
                                                 #hence the previous values of openDict is used (initialized as 0).

    del dict[oldPath] #Remove previous entry.

    for path in newPaths:
        
        if timeDict != 0: gValue = timeDict[path][-1] #If timeDict is initialized, g values have been computed for all paths in openDict,
                                                      #hence the final value of timeDict[path] is used as a starting g value for the new extended path.
        dict[path] = gValue #Update openDict with the new path using the same f value.

    return dict

def updateSPandTimeDict(SPDict, timeDict, oldPath):
    '''Updates SPDict and timeDict by removing the path which got expanded and will be replaced with its successors.'''

    del SPDict[oldPath]
    del timeDict[oldPath]

    return SPDict, timeDict

def expandPaths(G, destinationNode, dict):
    '''After a path is generated (its last node is connected with each one of its N neighbors creating N paths), it is expanded by adding the neighbor of its final node
    until a junction node (leads to multiple nodes) is met.
    
    :param G: A networkx Graph in which the problem is defined on.
    :param destinationNode: Char, the name of the final node (at the end of final segment).
    :param dict: Dictionary (openDict or successors) (key -> list of characters/nodes, value -> float), contains the path of openDict which will be expanded.
    :return: dict: Dictionary (key -> list of characters/nodes, value -> float), contains the expanded versions of paths in dict.
    :return: finalPath: Boolean, is True when one of the paths in paths is final (contains destinationNode).
    '''

    paths = list(dict.keys()) #Extract the list of paths from dict.

    finalPath = False #Initialize

    for path in paths:

        oldPath = path #Store the path before expansion, in order to update openDict.

        while(1): #Expand paths until a junction node is found
            junctions = 0
            for u,v,e in G.edges(data=True): #Iterate through the network, searching for nodes (v) to expand the path (u).
                if path[-1]==u and path[-1]!=destinationNode and v not in path: #If u and v are connected, u is not the final node and v is not already in the path (cycle)...
                    nextNode = v #Store v as the nextNode.
                    junctions += 1 #Increment the junction counter.

            if junctions != 1: #If u is connected to many nodes, it is junction and the process terminates.
                break
            else: #Else update the path.
                path = path + (nextNode,)
                
        dict[path] = dict.pop(oldPath) #Update dict with the expanded path.

        if destinationNode in path: finalPath = True #If path is final set finalPath to True.

    return dict, finalPath

def createVariables(G, openDict, initPos, destinationNode, goalV, goalPos):
    '''Before the computation of the speed profiles for each discovered path takes place, it is necessary to create the input variables needed for ALGORITHM 1 to operate on each path.
    This function creates and returns the lists maxSpeeds,positions,lengths,segmentIndices that describe the layout of each path.
    Furthermore it updates the goal state for each discovered path based on whether the destination node is included or not. 
    
    :param G: A networkx Graph in which the problem is defined on.
    :param openDict: Dictionary (key -> list of characters/nodes, value -> float), contains the f values of all the discovered paths.
    :param initPos: Tuple, the position of the init state (tuple -> list of segment indices + distance from end of segment).
    :param destinationNode: Character, the name of the final node (at the end of final segment).
    :param goalV: Float, the speed of the train at the goal state.
    :param goalPos: Tuple, the position of the goal state.
    :return: pathCharList: List of tuples, each entry contains the path information (maxSpeeds,positions,lengths,segmentIndices) in a tuple, for each path.
    :return: goals: List of tuples, the updated goal state for each path.
    :return: intermediate: Boolean, it is true if the path does not lead to the destination node in its current state, otherwise false.
    '''

    pathCharList, goals = [], [] #Initialize.

    for path in openDict.keys(): #Traverse all discovered paths.

        maxSpeeds, positions, lengths, segmentIndices = [], [], [], [] #Initialize the list that contain path information.

        for node in range(len(path)-1): #Traverse the nodes of the path.

            #Create segments of the path using pairs of consecutive nodes.
            segmentStart = path[node]
            segmentEnd = path[node+1]

            for u,v,e in G.edges(data=True): #Traverse the network.

                if u == segmentStart and v == segmentEnd: #Find the network edge defined by the path segment.

                    #Extract the path information.
                    maxSpeeds.append(e['maxspeed'])
                    lengths.append(e['length'])
                    segmentIndices.append(node+1)

                    #The positions list contains the cumulative distance traveled by the train.
                    #The entries are defined by the lengths of the segments with the exception of the 1st segment that is defined by the distance of head(T) from the end of the 1st segment.
                    if node == 0: positions.append(initPos[1]) 
                    else: positions.append(positions[node-1]+e['length']) 
    
        pathCharList.append((maxSpeeds,positions,lengths,segmentIndices)) #Store the path information as a tuple, for each path.

        #The goal state for each path differs from the goal state given in the problem's description.

        #The train is allowed to accelerate until the path's max speed if it is not a final one.
        if destinationNode not in path:
            goalV = maxSpeeds[-1]
            goalPos = ([segmentIndices[-1]],0)
            intermediate = True
        #If the path is a final one, the goal state for this path is updated to be the same as the one given in the problem's description.
        else:
            goalV = goalV
            goalPos = ([segmentIndices[-1]],goalPos[1])
            intermediate = False

        goal = (goalPos,goalV)
        goals.append(goal) #Store the goal state as a tuple, for each path.

    return pathCharList, goals, intermediate

def createSuccessors(current, openDict, oldPath):
    '''Creates successors dictionary that contains the expanded paths of current (path with the lowest f value).'''

    successors = {}

    for path in current:
        successors[path] = openDict[oldPath] #Paths of the successors dictionary are initialized with the g value of their immediate anchestor.

    return successors

def computeGValues(dict, init, initPos, trainChar, SPDict, timeDict, destinationNode, goalV, goalPos, finalPath, extented, debug, oldPath = 0):
    '''Computes the g values for each discovered path. The g value of a path is the time needed for the train to traverse it.
    
    :param dict: Dictionary (openDict or successors) (key -> list of characters/nodes, value -> float), contains the expanded paths the g values of which must be computed.
    :param init: Tuple, initial state of the problem.
    :param initPos: Tuple, the position of the init state (tuple -> list of segment indices + distance from end of segment).
    :param trainChar: Tuple, information describing the train (e.g. trainLength, trainMaxSpeed).
    :param SPDict: Dictionary (key -> list of characters/nodes, value -> list of states), contains the speed profiles computed for each path.
    :param timeDict: Dictionary (key -> list of characters/nodes, value -> list of floats), contains the time needed to reach a state in the speed profile of each path.
    :param destinationNode: Character, the name of the final node (at the end of final segment).
    :param goalV: Float, the speed of the train at the goal state.
    :param goalPos: Tuple, the position of the goal state.
    :param finalPath: Boolean, it is true when the path contains the final node.
    :param extended: Boolean, it is true when the path is an extension of a previous path.
    :param debug: Boolean, it is true when it is needed for debugging information to be printed.
    :param oldPath (default value = 0): List of strings, contains the path that was extented (except the first iteration in which there are no extensions and oldPath = 0).
    :return: dict: Dictionary (openDict or successors), contains the g values for the expanded paths.
    :return: SPDict: Dictionary, contains the speed profiles for the expanded paths.
    :return: timeDict: Boolean, contains the time needed to reach a state in the speed profile for each expanded path.
    '''

    pathCharList, goals, intermediate = createVariables(G, dict, initPos, destinationNode, goalV, goalPos) #Create the input variables needed for ALGORITHM 1 to operate on each path.
    variables = []

    for i in range(len(pathCharList)): #Traverse all discovered paths.

        currentPath = list(dict.keys())[i]
        if debug: print('Calculating g value on path ', currentPath)
        
        #The computation of g values differs based on the whether the path an extension of a previous one.
        #In all iteration besides the first one, the paths are extensions of previous paths.

        #If the path is not a extension (not extended) the variables needed for ALGORITHM's 1 operation,
        #are initialized using the init state given in the problem's description.
        if not extented: currentV, currentAcc, time, tSgm, hSgm, hExit, tExit, SP, flag, initNew, speedLimit, tMin, t1, t2, t3 = initVariables(init, goals[i], trainChar, pathCharList[i])
        #If the path is an extension the variables needed for ALGORITHM's 1 operation,
        #are initialized using the penultimate state in the speed profile of the path that is extented.
        #Instead of using the final state, the penultimate one is used in order to allow for ALGORITHM 1 to check whether a deceleration is needed
        #before starting the computation of the speed profile of the path's extension.
        #The reason behind is the fact that the speed limit of first segment of the extension is not known beforehand.
        #Hence it is quite possible that the train is not in a state (speed-wise) to enter the new segment and a deceleration is needed in earlier points of the path's speed profile (stored in SPDict).
        else:
            currentV, currentAcc, _, tSgm, hSgm, hExit, tExit, _, flag, initNew, speedLimit, tMin, t1, t2, t3 = initVariables(SPDict[oldPath][len(SPDict[oldPath])-2], goals[i], trainChar, pathCharList[i])
            time = timeDict[oldPath][0:len(timeDict[oldPath])-1]
            SP = SPDict[oldPath][0:len(timeDict[oldPath])-1]

        variables.append((currentV, currentAcc, time, tSgm, hSgm, hExit, tExit, SP, flag, initNew, speedLimit, tMin, t1, t2, t3)) #Store the initialized variables for each path.
        
        #If the path is a final one (contains the destination node), the updatePathChar function is applied.
        #which updates the path's description in accordance with the goal state 
        #(e.g. creates a new segment after the last one, with a speed limit equal to the desired speed of the train in goal).
        if finalPath: goals[i], pathCharList[i] = updatePathChar(goals[i], pathCharList[i]) 

        #Compute the speed profile for each path.
        SPDict[currentPath], timeDict[currentPath] = computeOptimalSP(goals[i], pathCharList[i], trainChar, variables[i], debug, intermediate)

        if timeDict[currentPath] != None:
            #Store the g value for each path.
            gValue = (timeDict[currentPath])[-1]
        else:
            #If no solution is found make the path a 'dead end' (infinite g value).
            SPDict[currentPath], timeDict[currentPath], gValue = [0], [np.inf], np.inf

        dict[currentPath] = gValue

    return dict, SPDict, timeDict

def createMap(G):
    '''Creates the mapDict dictionary that stores the straight-line distance of each node from the destination node.'''
    
    mapDict = {}
    
    for u,v,e in G.edges(data=True):
        if not str(u) in mapDict: mapDict[str(u)] = e['straight_line']
        if not str(v) in mapDict: mapDict[str(v)] = e['straight_line']

    return mapDict

def computeFValues(openDict, mapDict, maxStraightLineSpeed, debug):
    '''Computes the f values for each discovered path. The f value of a path is the time needed for the train to traverse it (g value)
    plus the estimation h using the heuristic function.
    
    :param openDict: Dictionary, contains the f values of all the discovered paths.
    :param mapDict: Dictionary (key -> character/node, value -> integers), stores the straight-line distance of each node from the destination node.
    :param maxStraightLineSpeed: integer, sets the speed value used in the heuristic function in order to make estimations.
    :return: openDictfValues: Dictionary (key -> list of characters/nodes, value -> floats), contains the f values for the expanded paths.'''

    openDictfValues = copy.deepcopy(openDict) #Instead of updating openDict with the f values, create a new copy of it in order to store the f values, maintain the g values for use in later steps.

    for path in openDict.keys(): #Traverse all paths in openDict.
        originName = path[-1] #Store the last node of the path.
        straightLineDistance = mapDict[originName] #Find the straight line distance from the last node.

        if debug:
            print('Calculating f value on path ', path)
            print('g value: ',openDictfValues[path])

        openDictfValues[path] = straightLineDistance/maxStraightLineSpeed + openDictfValues[path] #Compute and store the f value.

        if debug:
            print('heuristic: ', straightLineDistance/maxStraightLineSpeed)
            print('f value: ',openDictfValues[path])
        
    return openDictfValues

def computeShortestPath(G, variables, debug):
    '''ALGORITHM3: Using the problem's description provided by the createInputData function, as well as the network G,
    this algorithm provides the shortest path (time-wise) and its speed profile.'''

    #Initialization
    SPDict, timeDict = {}, {}
    mapDict = createMap(G)

    #Extract init and goal states, as well as the trainChar (trainLength etc.).
    _, initPos, init, goalV, goalPos, _, trainChar, _ = storeVariables(variables)

        #Extract train max speed.
    _, _, _, trainMaxSpeed = trainChar

    #Set the hyperparameter maxStraightLineSpeed.
    maxStraightLineSpeed = trainMaxSpeed

    #Initiliaze openDict and start path, find the final node.
    openDict, destinationNode, start = extractStartDestination(G, initPos, goalPos)

    #Generate the first paths using the start path.
    neighbors = generatePaths(G, start)
    if debug: print('Starting openDict: ', openDict)

    #Update openDict using the generated paths in neighbors.
    openDict = updateOpenDict(neighbors, openDict, start)
    if debug: print('openDict after path generation: ',openDict)

    #Expand the paths of openDict.
    openDict, finalPath = expandPaths(G, destinationNode, openDict)
    if debug: print('openDict after path expansion: ',openDict)

    #Compute g values for all paths in openDict.
    openDict, SPDict, timeDict = computeGValues(openDict, init, initPos, trainChar, SPDict, timeDict, destinationNode, goalV, goalPos, finalPath, False, debug)
    if debug: print('openDict with g values: ', openDict)

    #Compute f values for all paths in openDict.
    openDict_fValues = computeFValues(openDict, mapDict, maxStraightLineSpeed, debug)
    if debug:
        print('Data after f values computation.')
        print('openDict: ', openDict)
        print('SPDict: ', SPDict)
        print('timeDict: ', timeDict)

    while(openDict): #while openDict not empty
        
        current = min(openDict_fValues, key=openDict_fValues.get) #Find the path with lowest f value.
        oldPath = current #Store it before expanding it.

        current = generatePaths(G, current) #Generate new paths using the neighbors of its final nodes.
        successors = createSuccessors(current, openDict, oldPath) #Create the successors dictionary.
        if debug:
            print('\nCurrent fastest path: ',oldPath)
            print('Generated paths: ',successors.keys())

        #Expand the paths in successors.
        successors, finalPath = expandPaths(G, destinationNode, successors)
        if debug: print('Expanded paths: ',successors.keys())

        #Compute g values for all paths in successors.
        successors, SPDict, timeDict = computeGValues(successors, init, initPos, trainChar, SPDict, timeDict, destinationNode, goalV, goalPos, finalPath, True, debug, oldPath)

        #Update openDict, SPDict and timeDict using successors.
        openDict = updateOpenDict(successors, openDict, oldPath, timeDict)
        if debug: print('openDict with g values: ', openDict)
        SPDict, timeDict = updateSPandTimeDict(SPDict, timeDict, oldPath)

        #Compute f values for all paths in openDict.
        openDict_fValues = computeFValues(openDict, mapDict, maxStraightLineSpeed, debug)
        if debug:
            print('Data after f values computation.')
            print('openDict: ', openDict)
            print('SPDict: ', SPDict)
            print('timeDict: ', timeDict)

        #If a path containing the destination node is found, return the solution.
        if finalPath:
            current = min(successors, key=successors.get)
            return openDict, SPDict, current, timeDict

if __name__ == "__main__":

    # Initialize the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('Network_file', type=str)
    parser.add_argument('Characteristics_file', type=str)
    parser.add_argument('--debug', nargs='?', type=bool, const=True, default=False, help='Enable debug mode.')

    # Parse the arguments.
    args = parser.parse_args()

    # Read the variables from the file.
    variables = parseVariables(args.Characteristics_file)

    #Read the input data and create the network G.
    G = nx.read_edgelist(args.Network_file, create_using=nx.DiGraph)

    #Compute the solution.
    openDict, SPDict, path, timeDict = computeShortestPath(G, variables, args.debug)
    
    if openDict[path] != np.inf:
        print('Shortest path is: ',path)
        print('\nTotal time: ', openDict[path])
        print('\nOptimal Speed profile of the path\n')
        for i, (state, timeValue) in enumerate(zip(SPDict[path],timeDict[path]), start=1):
            print(f"State {i}: {state}, Time: {timeValue}")
    else:
        print('The problem cannot be solved!')
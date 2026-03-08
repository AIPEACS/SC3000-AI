# a list with 5 * 5 grid view.
# because of the display of matrix,
# it is supposed to be (0,0) at the bottom left corner, and (4,4) at the top right corner.
# x axis is from left to right, and y axis is from bottom to top.

# in the matrix, (0,0) is at the top left corner, and (4,4) is at the bottom right corner.
# x axis is the second argument, and y axis is the first argument.
# for example, map[2,1] is the position (1,2) in the grid view.

import random

start_point = (0,0)
end_point = (4,4)

road_blocking = [(2,1),(2,3)]

mess_up_probability = 0.2

def move_function_at_position(x,y,direction):
    
	# move up
	if direction == "u" and y < 4 and (x,y+1) not in road_blocking:
		return (x,y+1)
	# move down
	elif direction == "d" and y > 0 and (x,y-1) not in road_blocking:
		return (x,y-1)
	# move left
	elif direction == "l" and x > 0 and (x-1,y) not in road_blocking:
		return (x-1,y)
	# move right
	elif direction == "r" and x < 4 and (x+1,y) not in road_blocking:
		return (x+1,y)
	# if the moving direction is not available, stay at the same position
	else:
		return (x,y)

def move_perpendicular_to_direction(direction):
	if direction == "u" or direction == "d":
		return random.choice(["l","r"])
	elif direction == "l" or direction == "r":
		return random.choice(["u","d"])
	else:
		return None

def moving_function_with_messup_probability(x,y,direction,messup_probability):
	decider = random.random()
	if decider < 1- 2*messup_probability:
		return move_function_at_position(x,y,direction)
	elif decider >= 2 * messup_probability:
		return move_function_at_position(x,y,move_perpendicular_to_direction(direction))
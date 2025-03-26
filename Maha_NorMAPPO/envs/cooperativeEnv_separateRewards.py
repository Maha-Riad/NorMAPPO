from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses

import numpy as np
import pickle
import copy
import argparse
import sys

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import rendering
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# General Parameters
MAX_STEPS = 75
N_MAIL = 12
N_CITIZEN = 12
N_STREET = 12
N_VASE = 8

DELIVERY_ART = \
    ['FCS#############',
     '#     Z        #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '#  P           #',
     '#              #',
     '#              #',
     '#            L #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '################']

BACKGROUND_ART = \
    ['################',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '#              #',
     '################']


DELIVERY_FG_COLOURS = {' ': (870, 838, 678),  # Floor.
                        '#': (428, 135, 0),    # Walls.
                        'C': (0, 600, 67),     # Citizen.
                        'x': (850, 603, 270),  # Unused.
                        'P': (388, 400, 999),  # The player.
                        'L': (388, 400, 999),  # The second player. added by Maha
                        'Z': (388, 400, 999),  # The third player. added by Maha 
                        'F': (500, 0, 500),  # Mail.
                        'S': (999, 500, 0),      # Street.
                        'V': (900, 0, 20)}      # Vase.


def make_game(seed=None, demo=False):
    delivery_art = DELIVERY_ART
    what_lies_beneath = BACKGROUND_ART
    sprites = {'P': PlayerSprite}
    sprites ['L']= Player2Sprite # added by Maha
    sprites ['Z']= Player3Sprite # added by Maha

    if demo:
        raise NotImplementedError
    else:
        drapes = {'X': JudgeDrape}

    drapes['F'] = MailDrape
    drapes['C'] = CitizenDrape
    drapes['S'] = StreetDrape
    drapes['V'] = VaseDrape

    update_schedule = [['F'],
                       ['C'],
                       ['S'],
                       ['V'],
                       ['X'],
                       ['L'], #added by Maha
                       ['Z'], #added by Maha
                       ['P']]


    return ascii_art.ascii_art_to_game(
        delivery_art, what_lies_beneath, sprites, drapes,
        update_schedule=update_schedule)

def scalar_to_idx(x):
    row = x%14
    col = int(np.floor(x/14))
    return (row+1, col+1)


class MailDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(MailDrape, self).__init__(curtain, character)
        self.curtain.fill(False)

    def update(self, actions, board, layers, backdrop, things, the_plot): #MailDrape
        del backdrop

        #added by Maha
        if isinstance(actions, (list, tuple, np.ndarray)):
            if actions.all()!=None:
                p_action=actions[0]
                l_action=actions[1]
                z_action=actions[2]
            else:
                p_action=None
                l_action=None
                z_action=None
        else:
                p_action=actions
                l_action=None
                z_action=None
        #End of Maha's Addition
        if the_plot.frame == 0:
            # Random initialization of player, fire and citizen
            random_positions = np.random.choice(14*14, size=N_MAIL+N_CITIZEN+N_STREET+N_VASE+1+1+1, replace=False) # added two 1's my Maha to represent the positions of the added players
            for i in range(N_MAIL):
                tmp_idx = scalar_to_idx(random_positions[i])
                self.curtain[tmp_idx] = True
            the_plot['P_pos'] = scalar_to_idx(random_positions[-3])
            the_plot['L_pos'] = scalar_to_idx(random_positions[-2]) #added by Maha
            the_plot['Z_pos'] = scalar_to_idx(random_positions[-1]) #added by Maha
            the_plot['C_pos'] = [scalar_to_idx(i) for i in random_positions[N_MAIL:N_MAIL+N_CITIZEN]]
            the_plot['S_pos'] = [scalar_to_idx(i) for i in random_positions[N_MAIL+N_CITIZEN
                                                                            :N_MAIL+N_CITIZEN+N_STREET]]
            the_plot['V_pos'] = [scalar_to_idx(i) for i in random_positions[N_MAIL+N_CITIZEN+N_STREET
                                                                            :N_MAIL+N_CITIZEN+N_STREET+N_VASE]]

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        player2_pattern_position = things['L'].position #added by Maha
        player2_row = player2_pattern_position.row #added by Maha
        player2_col = player2_pattern_position.col #added by Maha

        player3_pattern_position = things['Z'].position #added by Maha
        player3_row = player3_pattern_position.row #added by Maha
        player3_col = player3_pattern_position.col #added by Maha

        # Check for 'pick up' action:
        if p_action == 5 and self.curtain[(player_row-1, player_col)]: # grab upward?
            self.curtain[(player_row-1, player_col)] = False
            the_plot.add_reward(np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        if p_action == 6 and self.curtain[(player_row+1, player_col)]: # grab downward?
            self.curtain[(player_row+1, player_col)] = False
            the_plot.add_reward(np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        if p_action == 7 and self.curtain[(player_row, player_col-1)]: # grab leftward?
            self.curtain[(player_row, player_col-1)] = False
            the_plot.add_reward(np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        if p_action == 8 and self.curtain[(player_row, player_col+1)]: # grab rightward?
            self.curtain[(player_row, player_col+1)] = False
            the_plot.add_reward(np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))

        #The below if statements are added by Maha
        if l_action == 5 and self.curtain[(player2_row-1, player2_col)]: # grab upward?
            self.curtain[(player2_row-1, player2_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]))
        if l_action == 6 and self.curtain[(player2_row+1, player2_col)]: # grab downward?
            self.curtain[(player2_row+1, player2_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]))
        if l_action == 7 and self.curtain[(player2_row, player2_col-1)]: # grab leftward?
            self.curtain[(player2_row, player2_col-1)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]))
        if l_action == 8 and self.curtain[(player2_row, player2_col+1)]: # grab rightward?
            self.curtain[(player2_row, player2_col+1)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]))

        #The below if statements are added by Maha
        if z_action == 5 and self.curtain[(player3_row-1, player3_col)]: # grab upward?
            self.curtain[(player3_row-1, player3_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]))
        if z_action == 6 and self.curtain[(player3_row+1, player3_col)]: # grab downward?
            self.curtain[(player3_row+1, player3_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]))
        if z_action == 7 and self.curtain[(player3_row, player3_col-1)]: # grab leftward?
            self.curtain[(player3_row, player3_col-1)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]))
        if z_action == 8 and self.curtain[(player3_row, player3_col+1)]: # grab rightward?
            self.curtain[(player3_row, player3_col+1)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]))

        #if self.curtain[player_pattern_position]:
        #    the_plot.add_reward(np.array([1, 0]))
        #    self.curtain[player_pattern_position] = False

class CitizenDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(CitizenDrape, self).__init__(curtain, character)
        self.curtain.fill(False)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop

        #added by Maha
        if isinstance(actions, (list, tuple, np.ndarray)):
            if actions.all()!=None:
                p_action=actions[0]
                l_action=actions[1]
                z_action=actions[2]
            else:
                p_action=None
                l_action=None
                z_action=None
        else:
                p_action=actions
                l_action=None
                z_action=None
        #End of Maha's Addition

        if the_plot.frame == 0:
            # Random initialization of player and fire
            citizen_positions = the_plot['C_pos']
            for pos in citizen_positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        player2_pattern_position = things['L'].position #added by Maha
        player2_row = player2_pattern_position.row #added by Maha
        player2_col = player2_pattern_position.col #added by Maha

        player3_pattern_position = things['Z'].position #added by Maha
        player3_row = player3_pattern_position.row #added by Maha
        player3_col = player3_pattern_position.col #added by Maha



        # Check for 'pick up' action:
        if p_action == 5 and self.curtain[(player_row-1, player_col)]: # grab upward?
            self.curtain[(player_row-1, player_col)] = False
            the_plot.add_reward(np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        if p_action == 6 and self.curtain[(player_row+1, player_col)]: # grab downward?
            self.curtain[(player_row+1, player_col)] = False
            the_plot.add_reward(np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        if p_action == 7 and self.curtain[(player_row, player_col-1)]: # grab leftward?
            self.curtain[(player_row, player_col-1)] = False
            the_plot.add_reward(np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        if p_action == 8 and self.curtain[(player_row, player_col+1)]: # grab rightward?
            self.curtain[(player_row, player_col+1)] = False
            the_plot.add_reward(np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))

         # added by Maha
        if l_action == 5 and self.curtain[(player2_row-1, player2_col)]: # grab upward?
            self.curtain[(player2_row-1, player2_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]))
        if l_action == 6 and self.curtain[(player2_row+1, player2_col)]: # grab downward?
            self.curtain[(player2_row+1, player2_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]))
        if l_action == 7 and self.curtain[(player2_row, player2_col-1)]: # grab leftward?
            self.curtain[(player2_row, player2_col-1)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]))
        if l_action == 8 and self.curtain[(player2_row, player2_col+1)]: # grab rightward?
            self.curtain[(player2_row, player2_col+1)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]))


         # added by Maha
        if z_action == 5 and self.curtain[(player3_row-1, player3_col)]: # grab upward?
            self.curtain[(player3_row-1, player3_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]))
        if z_action == 6 and self.curtain[(player3_row+1, player3_col)]: # grab downward?
            self.curtain[(player3_row+1, player3_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]))
        if z_action == 7 and self.curtain[(player3_row, player3_col-1)]: # grab leftward?
            self.curtain[(player3_row, player3_col-1)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]))
        if z_action == 8 and self.curtain[(player3_row, player3_col+1)]: # grab rightward?
            self.curtain[(player3_row, player3_col+1)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]))

class StreetDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(StreetDrape, self).__init__(curtain, character)
        self.curtain.fill(False)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop
        
         #added by Maha
        if isinstance(actions, (list, tuple, np.ndarray)):
            if actions.all()!=None:
                p_action=actions[0]
                l_action=actions[1]
                z_action=actions[2]
            else:
                p_action=None
                l_action=None
                z_action=None
        else:
                p_action=actions
                l_action=None
                z_action=None
        #End of Maha's Addition

        if the_plot.frame == 0:
            # Random initialization of player and fire
            citizen_positions = the_plot['S_pos']
            for pos in citizen_positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        player2_pattern_position = things['L'].position #added by Maha
        player2_row = player2_pattern_position.row #added by Maha
        player2_col = player2_pattern_position.col #added by Maha

        player3_pattern_position = things['Z'].position #added by Maha
        player3_row = player3_pattern_position.row #added by Maha
        player3_col = player3_pattern_position.col #added by Maha



        # Check for 'pick up' action:
        if p_action == 5 and self.curtain[(player_row-1, player_col)]: # grab upward?
            self.curtain[(player_row-1, player_col)] = False
            the_plot.add_reward(np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        if p_action == 6 and self.curtain[(player_row+1, player_col)]: # grab downward?
            self.curtain[(player_row+1, player_col)] = False
            the_plot.add_reward(np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        if p_action == 7 and self.curtain[(player_row, player_col-1)]: # grab leftward?
            self.curtain[(player_row, player_col-1)] = False
            the_plot.add_reward(np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        if p_action == 8 and self.curtain[(player_row, player_col+1)]: # grab rightward?
            self.curtain[(player_row, player_col+1)] = False
            the_plot.add_reward(np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))

         # added by Maha
        if l_action == 5 and self.curtain[(player2_row-1, player2_col)]: # grab upward?
            self.curtain[(player2_row-1, player2_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]))
        if l_action == 6 and self.curtain[(player2_row+1, player2_col)]: # grab downward?
            self.curtain[(player2_row+1, player2_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]))
        if l_action == 7 and self.curtain[(player2_row, player2_col-1)]: # grab leftward?
            self.curtain[(player2_row, player2_col-1)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]))
        if l_action == 8 and self.curtain[(player2_row, player2_col+1)]: # grab rightward?
            self.curtain[(player2_row, player2_col+1)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]))

        # added by Maha
        if z_action == 5 and self.curtain[(player3_row-1, player3_col)]: # grab upward?
            self.curtain[(player3_row-1, player3_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]))
        if z_action == 6 and self.curtain[(player3_row+1, player3_col)]: # grab downward?
            self.curtain[(player3_row+1, player3_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]))
        if z_action == 7 and self.curtain[(player3_row, player3_col-1)]: # grab leftward?
            self.curtain[(player3_row, player3_col-1)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]))
        if z_action == 8 and self.curtain[(player3_row, player3_col+1)]: # grab rightward?
            self.curtain[(player3_row, player3_col+1)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]))

class VaseDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(VaseDrape, self).__init__(curtain, character)
        self.curtain.fill(False)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop
         #added by Maha
        if isinstance(actions, (list, tuple, np.ndarray)):
            if actions.all()!=None:
                p_action=actions[0]
                l_action=actions[1]
                z_action=actions[2]
            else:
                p_action=None
                l_action=None
                z_action=None
        else:
                p_action=actions
                l_action=None
                z_action=None
        #End of Maha's Addition

        if the_plot.frame == 0:
            # Random initialization of player and fire
            citizen_positions = the_plot['V_pos']
            for pos in citizen_positions:
                self.curtain[pos] = True

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        player2_pattern_position = things['L'].position #added by Maha
        player2_row = player2_pattern_position.row #added by Maha
        player2_col = player2_pattern_position.col #added by Maha

        player3_pattern_position = things['Z'].position #added by Maha
        player3_row = player3_pattern_position.row #added by Maha
        player3_col = player3_pattern_position.col #added by Maha

        if self.curtain[(player_row, player_col)]:
            self.curtain[(player_row, player_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0.]))
        #added by Maha
        if self.curtain[(player2_row, player2_col)]:
            self.curtain[(player2_row, player2_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0.]))
        #added by Maha
        if self.curtain[(player3_row, player3_col)]:
            self.curtain[(player3_row, player3_col)] = False
            the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.]))


class HouseDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(HouseDrape, self).__init__(curtain, character)
        self.curtain.fill(False)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop

        if the_plot.frame == 0:
            # Random initialization of player and fire
            citizen_positions = the_plot['H_pos']
            for pos in citizen_positions:
                self.curtain[pos] = True


class PlayerSprite(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#H.')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused.
        #added by Maha
        if isinstance(actions, (list, tuple, np.ndarray)):
            if actions.all()!=None:
                p_action=actions[0]
             
            else:
                p_action=None
        else:
                p_action=actions
        #End of Maha's Addition

        if the_plot.frame == 0:
            self._teleport(the_plot['P_pos'])

        if p_action == 0:    # go upward?
            self._north(board, the_plot)
        elif p_action == 1:  # go downward?
            self._south(board, the_plot)
        elif p_action == 2:  # go leftward?
            self._west(board, the_plot)
        elif p_action == 3:  # go rightward?
            self._east(board, the_plot)

class Player2Sprite(prefab_sprites.MazeWalker): #added by Maha

    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(Player2Sprite, self).__init__(
        corner, position, character, impassable='#H.')

    def update(self, actions, board, layers, backdrop, things, the_plot):
    #def update(self, actions, L_actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused.

        #added by Maha
        if isinstance(actions, (list, tuple, np.ndarray)):
            if actions.all()!=None:
                l_action=actions[1]
            else:
                l_action=None
        else:
                l_action=None
        #End of Maha's Addition

        if the_plot.frame == 0:
            self._teleport(the_plot['L_pos'])

        if l_action == 0:    # go upward?
            self._north(board, the_plot)
        elif l_action == 1:  # go downward?
            self._south(board, the_plot)
        elif l_action == 2:  # go leftward?
            self._west(board, the_plot)
        elif l_action == 3:  # go rightward?
            self._east(board, the_plot)



class Player3Sprite(prefab_sprites.MazeWalker): #added by Maha

    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(Player3Sprite, self).__init__(
        corner, position, character, impassable='#H.')

    def update(self, actions, board, layers, backdrop, things, the_plot):
    #def update(self, actions, L_actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused.
       
        #added by Maha
        if isinstance(actions, (list, tuple, np.ndarray)):
            if actions.all()!=None:
                z_action=actions[2]
            else:
                z_action=None
        else:
                z_action=None
        #End of Maha's Addition

        if the_plot.frame == 0:
            self._teleport(the_plot['Z_pos'])

        if z_action == 0:    # go upward?
            self._north(board, the_plot)
        elif z_action == 1:  # go downward?
            self._south(board, the_plot)
        elif z_action == 2:  # go leftward?
            self._west(board, the_plot)
        elif z_action == 3:  # go rightward?
            self._east(board, the_plot)


class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)
        self._step_counter = 0
        self._max_steps = MAX_STEPS


    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Clear our curtain and mark the locations of all the boxes True.
        self.curtain.fill(False)
        the_plot.add_reward(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        #the_plot.add_reward(-0.1)
        self._step_counter += 1

        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.
        if isinstance(actions, (list, tuple, np.ndarray)):
            if (actions.any() == 9) or(self._step_counter == self._max_steps) :
                the_plot.terminate_episode()
        else:
            if (actions == 9) or (self._step_counter == self._max_steps):
                the_plot.terminate_episode()


def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)


    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                         'w': 5,
                         's': 6,
                         'a': 7,
                         'd': 8,
                         -1: 4,
                         'q': 9, 'Q': 9},
        delay=1000,
        colour_fg=DELIVERY_FG_COLOURS)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo', help='Record demonstrations',
                        action='store_true')
    args = parser.parse_args()
    if args.demo:
        main(demo=True)
    else:
        main(demo=False)

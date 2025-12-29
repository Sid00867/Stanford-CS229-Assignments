from __future__ import division, print_function
from math import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CartPole:
    def __init__(self, physics):
        self.physics = physics
        self.mass_cart = 1.0
        self.mass_pole = 0.3
        self.mass = self.mass_cart + self.mass_pole
        self.length = 0.7  # actually half the pole length
        self.pole_mass_length = self.mass_pole * self.length

        # Attributes for interactive plotting
        self.fig = None
        self.ax = None
        self.pole_line = None
        self.cart_patch = None
        self.axle_patch = None
        self.title = None

    def simulate(self, action, state_tuple):
        """
        Simulation dynamics of the cart-pole system (No changes needed here)
        """
        x, x_dot, theta, theta_dot = state_tuple
        costheta, sintheta = cos(theta), sin(theta)
        force = self.physics.force_mag if action > 0 else (-1 * self.physics.force_mag)
        temp = (force + self.pole_mass_length * theta_dot * theta_dot * sintheta) / self.mass
        theta_acc = (self.physics.gravity * sintheta - temp * costheta) / (self.length * (4/3 - self.mass_pole * costheta * costheta / self.mass))
        x_acc = temp - self.pole_mass_length * theta_acc * costheta / self.mass
        new_x = x + self.physics.tau * x_dot
        new_x_dot = x_dot + self.physics.tau * x_acc
        new_theta = theta + self.physics.tau * theta_dot
        new_theta_dot = theta_dot + self.physics.tau * theta_acc
        return (new_x, new_x_dot, new_theta, new_theta_dot)

    def get_state(self, state_tuple):
        """
        Discretizes the continuous state vector. (No changes needed here)
        """
        x, x_dot, theta, theta_dot = state_tuple
        one_deg, six_deg, twelve_deg, fifty_deg = pi/180, 6*pi/180, 12*pi/180, 50*pi/180
        total_states = 163
        state = 0
        if x < -2.4 or x > 2.4 or theta < -twelve_deg or theta > twelve_deg:
            return total_states - 1
        else:
            if x < -1.5: state = 0
            elif x < 1.5: state = 1
            else: state = 2
            
            if x_dot < -0.5: pass
            elif x_dot < 0.5: state += 3
            else: state += 6
            
            if theta < -six_deg: pass
            elif theta < -one_deg: state += 9
            elif theta < 0: state += 18
            elif theta < one_deg: state += 27
            elif theta < six_deg: state += 36
            else: state += 45
            
            if theta_dot < -fifty_deg: pass
            elif theta_dot < fifty_deg: state += 54
            else: state += 108
        return state

    def init_display(self):
        """
        Initializes the matplotlib figure for animation.
        Call this ONCE before your simulation loop.
        """
        self.fig, self.ax = plt.subplots(1)
        plt.ion()  # Turn on interactive mode
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-0.5, 3.5)
        
        # Create plot objects that will be updated later
        self.pole_line, = self.ax.plot([], [], 'k-', lw=3) # Note the comma!
        self.cart_patch = patches.Rectangle((0 - 0.4, -0.25), 0.8, 0.25,
                                           linewidth=1, edgecolor='k', facecolor='cyan')
        self.axle_patch = patches.Rectangle((0 - 0.01, -0.5), 0.02, 0.25,
                                           linewidth=1, edgecolor='k', facecolor='r')
        self.ax.add_patch(self.cart_patch)
        self.ax.add_patch(self.axle_patch)
        self.title = self.ax.set_title('Starting...')
        plt.show()

    def show_cart(self, state_tuple, pause_time):
        """
        Updates the cart-pole display in an existing window.
        """
        # Ensure the display has been initialized
        if self.fig is None:
            self.init_display()

        x, _, theta, theta_dot = state_tuple
        
        # Update pole data
        pole_x = [x, x + 4 * self.length * sin(theta)]
        pole_y = [0, 4 * self.length * cos(theta)]
        self.pole_line.set_data(pole_x, pole_y)

        # Update cart's position
        self.cart_patch.set_xy((x - 0.4, -0.25))
        self.axle_patch.set_xy((x - 0.01, -0.5))

        # Update title with current state info
        x_dot_str, theta_str, theta_dot_str = '\\dot{x}', '\\theta', '\\dot{\\theta}'
        title_text = f'x: {state_tuple[0]:.3f}, ${x_dot_str}$: {state_tuple[1]:.3f}, '
        title_text += f'${theta_str}$: {state_tuple[2]:.3f}, ${theta_dot_str}$: {state_tuple[3]:.3f}'
        self.title.set_text(title_text)

        # Pause to allow the canvas to redraw
        plt.pause(pause_time)

class Physics:
    gravity = 9.8
    force_mag = 10.0
    tau = 0.02  # seconds between state updates
import sys
import random
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, 
    QHBoxLayout, QLabel, QMessageBox, QSpinBox, QGridLayout
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QColor, QFont


class MicrotubuleNetworkWidget(QWidget):
    """
    A widget that displays a 'network of microtubules' as a 2D grid of squares.
    Each row represents one microtubule; each column represents a tubulin unit.
    Color scheme (toy example):
        - +1 (or 1)  = Yellow
        - -1 (or 0)  = Light Blue
    """
    
    def __init__(self, num_microtubules=4, dimers_per_microtubule=8):
        super().__init__()
        self.num_microtubules = num_microtubules
        self.dimers_per_microtubule = dimers_per_microtubule
        
        # The total number of "units" in the network
        self.total_units = self.num_microtubules * self.dimers_per_microtubule
        
        # Initialize the states as an array of +1 or -1
        # We start them randomly
        self.states = np.random.choice([-1, 1], size=(self.num_microtubules, self.dimers_per_microtubule))
        
        # Placeholder for the weight matrix
        self.W = np.zeros((self.total_units, self.total_units))
        
        # Create some example memory patterns (toy examples).
        # Each pattern is shaped (num_microtubules, dimers_per_microtubule).
        # We'll flatten them to learn in a Hopfield manner.
        pattern1 = np.ones((self.num_microtubules, self.dimers_per_microtubule))
        pattern2 = np.ones((self.num_microtubules, self.dimers_per_microtubule)) * -1
        
        # For variety, let's invert a few random elements in pattern2
        for _ in range(int(self.total_units * 0.3)):
            i_rand = np.random.randint(0, self.num_microtubules)
            j_rand = np.random.randint(0, self.dimers_per_microtubule)
            pattern2[i_rand, j_rand] *= -1
        
        # Store these patterns for demonstration
        # You can add more or define them in code to represent “memories”
        self.memory_patterns = [pattern1, pattern2]
        
        # Build the weight matrix from these memory patterns
        self.build_weight_matrix(self.memory_patterns)
        
        # For convenient painting, define a minimum size
        self.setMinimumSize(600, 400)
    
    def build_weight_matrix(self, patterns):
        """
        Build a simplified Hopfield-like weight matrix from the provided memory patterns.
        W_ij = sum_over_p( x_i^p * x_j^p ) for all patterns p, with zero diagonal.
        
        We flatten each pattern to shape (total_units,) for the standard Hopfield rule.
        """
        self.W = np.zeros((self.total_units, self.total_units))
        
        for p in patterns:
            x = p.flatten()  # shape (total_units,)
            # Outer product to update weights
            self.W += np.outer(x, x)
        
        # Zero out the diagonal to avoid self-reinforcement
        np.fill_diagonal(self.W, 0)
        
        # Optionally normalize or scale if desired; for demonstration, we won’t do that here
    
    def paintEvent(self, event):
        """
        Draw the microtubule network states.
         - +1 => Yellow
         - -1 => Light Blue
        """
        super().paintEvent(event)
        painter = QPainter(self)
        
        margin = 10
        spacing = 2
        
        width_avail = self.width() - 2 * margin
        height_avail = self.height() - 2 * margin
        
        # We'll draw the states in a grid: rows = num_microtubules, cols = dimers_per_microtubule
        # Let's find a suitable square size
        max_square_w = width_avail // self.dimers_per_microtubule
        max_square_h = height_avail // self.num_microtubules
        
        square_size = min(max_square_w, max_square_h)
        
        for i in range(self.num_microtubules):
            for j in range(self.dimers_per_microtubule):
                rect_x = margin + j * square_size
                rect_y = margin + i * square_size
                state = self.states[i, j]
                
                if state == 1:
                    painter.setBrush(QColor("yellow"))
                else:
                    # state = -1
                    painter.setBrush(QColor("lightblue"))
                
                painter.drawRect(rect_x, rect_y, square_size - spacing, square_size - spacing)
        
        painter.end()
    
    def randomize_states(self):
        """
        Randomly choose +1 or -1 for each tubulin unit.
        """
        self.states = np.random.choice([-1, 1], size=(self.num_microtubules, self.dimers_per_microtubule))
        self.update()
    
    def partial_pattern(self, pattern_idx=0, noise_level=0.2):
        """
        Initialize the network states to one of the stored memory patterns, 
        then add random flips as 'noise' to demonstrate partial recall.
        
        :param pattern_idx: which pattern in self.memory_patterns to use
        :param noise_level: fraction of units to flip randomly
        """
        if pattern_idx < 0 or pattern_idx >= len(self.memory_patterns):
            return
        
        chosen_pattern = self.memory_patterns[pattern_idx].copy()
        total_units = self.total_units
        num_flips = int(noise_level * total_units)
        
        # Flatten, flip some random positions, then reshape
        flat_pattern = chosen_pattern.flatten()
        flip_indices = np.random.choice(np.arange(total_units), size=num_flips, replace=False)
        for idx in flip_indices:
            flat_pattern[idx] *= -1
        
        self.states = flat_pattern.reshape((self.num_microtubules, self.dimers_per_microtubule))
        self.update()
    
    def step_update(self):
        """
        Perform a single synchronous update of the entire network 
        using the Hopfield update rule:
        
            s_i(t+1) = sign( sum_j( W_ij * s_j(t) ) )
            
        We'll flatten the states for matrix multiplication, then reshape.
        """
        flat_states = self.states.flatten()  # shape (total_units,)
        
        # Synchronous update
        new_states = self.W @ flat_states  # matrix multiplication
        # Apply sign, replace 0 with +1 if needed
        new_states = np.sign(new_states)
        new_states[new_states == 0] = 1  # In rare cases of exact zero, pick +1 arbitrarily
        
        self.states = new_states.reshape((self.num_microtubules, self.dimers_per_microtubule))
        self.update()
    
    def run_until_stable(self, max_iterations=50):
        """
        Run synchronous updates until the network no longer changes or we hit max_iterations.
        """
        for _ in range(max_iterations):
            old_states = self.states.copy()
            self.step_update()
            
            # If no change occurred, we consider it stable
            if np.array_equal(old_states, self.states):
                break


class MainWindow(QMainWindow):
    """
    Main window that holds the MicrotubuleNetworkWidget and provides buttons 
    to demonstrate memory storing, partial pattern initialization, update steps, etc.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network of Microtubules - Memory Switching (Toy Model)")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.network_widget = MicrotubuleNetworkWidget(num_microtubules=4, dimers_per_microtubule=8)
        
        # Layout
        v_layout = QVBoxLayout()
        
        # Button row
        btn_layout = QHBoxLayout()
        
        # Randomize states
        self.btn_random = QPushButton("Randomize States")
        self.btn_random.clicked.connect(self.network_widget.randomize_states)
        btn_layout.addWidget(self.btn_random)
        
        # Load partial pattern 1
        self.btn_pattern1 = QPushButton("Load Partial Memory #1")
        self.btn_pattern1.clicked.connect(lambda: self.network_widget.partial_pattern(0, noise_level=0.3))
        btn_layout.addWidget(self.btn_pattern1)
        
        # Load partial pattern 2
        self.btn_pattern2 = QPushButton("Load Partial Memory #2")
        self.btn_pattern2.clicked.connect(lambda: self.network_widget.partial_pattern(1, noise_level=0.3))
        btn_layout.addWidget(self.btn_pattern2)
        
        # Single update
        self.btn_step = QPushButton("Step Update")
        self.btn_step.clicked.connect(self.network_widget.step_update)
        btn_layout.addWidget(self.btn_step)
        
        # Run until stable
        self.btn_run = QPushButton("Run Until Stable")
        self.btn_run.clicked.connect(self.run_until_stable_handler)
        btn_layout.addWidget(self.btn_run)
        
        # Assemble
        v_layout.addWidget(self.network_widget)
        v_layout.addLayout(btn_layout)
        central_widget.setLayout(v_layout)
        
        self.resize(900, 600)
    
    def run_until_stable_handler(self):
        self.network_widget.run_until_stable()
        QMessageBox.information(self, "Converged", "The network has converged (or max iterations reached).")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

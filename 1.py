import sys
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, 
    QHBoxLayout, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QPainter, QColor, QFont

class MicrotubuleWidget(QWidget):
    """
    A QWidget that draws a simplified 'microtubule' made up of tubulin dimers.
    Each dimer has a 'quantum state' that collapses upon measurement.
    """
    
    def __init__(self, num_dimers=16):
        super().__init__()
        
        # Number of tubulin dimers to simulate
        self.num_dimers = num_dimers
        
        # We'll model each dimer as a 2-state system: 0 or 1.
        # Start each in an 'undecided' superposition indicated by None
        self.states = [None] * self.num_dimers
        
        # Define a "memory pattern" we consider the 'correct' or 'target' pattern
        # You can change this to any combination of 0s and 1s
        self.memory_pattern = [0, 1, 1, 0, 1, 0, 1, 1,
                               1, 0, 0, 1, 1, 1, 0, 1]
        
        self.setMinimumSize(600, 200)
    
    def paintEvent(self, event):
        """
        Paints the tubulin states.
        - Gray if still in superposition (None).
        - Blue if the state is 0.
        - Yellow if the state is 1.
        """
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Determine the size of each square
        margin = 10
        available_width = self.width() - 2 * margin
        available_height = self.height() - 2 * margin
        
        # We'll draw the dimers in a single row, so let's compute square size
        # based on the maximum feasible space
        square_size = min(available_width // self.num_dimers, available_height)
        
        # Starting x for drawing
        start_x = margin
        start_y = (self.height() - square_size) // 2
        
        for i in range(self.num_dimers):
            rect = QRect(start_x + i*square_size, start_y, square_size, square_size)
            
            state = self.states[i]
            if state is None:
                # Superposition
                painter.setBrush(QColor("lightgray"))
            elif state == 0:
                painter.setBrush(QColor("lightblue"))
            else:  # state == 1
                painter.setBrush(QColor("yellow"))
            
            painter.drawRect(rect)
            
        painter.end()
    
    def randomize_superposition(self):
        """Randomly choose states between 0 and 1 as if in an uncollapsed superposition."""
        self.states = [None] * self.num_dimers
        self.update()
    
    def collapse_states(self):
        """
        Simulates measurement:
        - For each dimer in superposition (None), pick 0 or 1 at random.
        - If already 0 or 1, keep it as is.
        """
        for i in range(self.num_dimers):
            if self.states[i] is None:
                self.states[i] = random.randint(0, 1)
        self.update()

    def check_memory(self):
        """Check if the collapsed pattern matches the target memory pattern."""
        if None in self.states:
            # Some states still haven't collapsed
            return False
        
        return self.states == self.memory_pattern


class MainWindow(QMainWindow):
    """
    Main application window that holds the MicrotubuleWidget and some buttons.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Microtubule Memory (Toy Simulation)")
        
        # Create the central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.microtubule_widget = MicrotubuleWidget(num_dimers=16)
        
        # Layout
        v_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        
        # Buttons
        self.btn_randomize = QPushButton("Randomize Superposition")
        self.btn_randomize.clicked.connect(self.microtubule_widget.randomize_superposition)
        
        self.btn_collapse = QPushButton("Collapse State")
        self.btn_collapse.clicked.connect(self.handle_collapse)
        
        # Add to layout
        btn_layout.addWidget(self.btn_randomize)
        btn_layout.addWidget(self.btn_collapse)
        
        v_layout.addWidget(self.microtubule_widget)
        v_layout.addLayout(btn_layout)
        
        central_widget.setLayout(v_layout)
        
        self.resize(800, 300)
    
    def handle_collapse(self):
        """
        Collapse the microtubule states, then check if the resulting pattern 
        matches the 'memory' pattern.
        """
        self.microtubule_widget.collapse_states()
        
        # Check if the final pattern matches the memory pattern
        if self.microtubule_widget.check_memory():
            QMessageBox.information(self, "Memory Check", 
                "Congratulations! The collapsed pattern matches the memory.")
        else:
            QMessageBox.information(self, "Memory Check", 
                "The collapsed pattern does not match the memory.")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

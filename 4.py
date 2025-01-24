import sys
import random
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, 
    QHBoxLayout, QLabel
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor


# ----------------------------------------------------------------------
#                           Hopfield Functions
# ----------------------------------------------------------------------

def bits_to_pm1(bit_array):
    """
    Convert a 1D or 2D array of bits {0, 1} to {+1, -1}.
    """
    return np.where(bit_array == 1, 1, -1)

def pm1_to_bits(pm1_array):
    """
    Convert a 1D or 2D array of {+1, -1} to bits {0, 1}.
    """
    return np.where(pm1_array == 1, 1, 0)

def build_hopfield_weights(pattern_pm1):
    """
    Build a Hopfield weight matrix W from a single pattern in {+1, -1}.
    Using the standard outer-product rule, with zero diagonal.
    
    pattern_pm1 is shape (n,) or (rows*cols,) in {+1, -1}.
    """
    length = pattern_pm1.size
    W = np.outer(pattern_pm1, pattern_pm1)
    # Zero out diagonal
    np.fill_diagonal(W, 0)
    return W

def hopfield_converge(initial_state_pm1, W, max_iter=20):
    """
    Synchronously update the Hopfield network until stable or max_iter.
    initial_state_pm1: shape (n,) in {+1, -1}.
    W: weight matrix of shape (n, n).
    """
    s = initial_state_pm1.copy()
    for _ in range(max_iter):
        old_s = s.copy()
        # Synchronous update
        s = W @ s  # matrix multiplication
        s = np.sign(s)  # Convert to +1 or -1
        # If no change, we're stable
        if np.array_equal(s, old_s):
            break
    return s


# ----------------------------------------------------------------------
#             Genetic Algorithm for Evolving Bit Patterns
# ----------------------------------------------------------------------

class HopfieldGA:
    def __init__(self, rows=8, cols=8, population_size=20, mutation_rate=0.02):
        self.rows = rows
        self.cols = cols
        self.n = rows * cols
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        # Create a random "target" bit pattern
        # (You could define a custom pattern if you like.)
        self.target_bits = np.random.randint(0, 2, size=(rows, cols))
        
        # We'll build the Hopfield weight matrix from this target pattern
        target_pm1 = bits_to_pm1(self.target_bits.flatten())
        self.W = build_hopfield_weights(target_pm1)  # shape (n, n)
        
        # Initialize population: each individual is shape (n,) in {0, 1}
        self.population = [
            np.random.randint(0, 2, size=self.n) for _ in range(population_size)
        ]
        
        self.best_individual = None  # shape (n,)
        self.best_fitness = -1
    
    def evaluate_fitness(self, individual):
        """
        Evaluate how close the Hopfield network's stable state is to the target pattern.
        
        Steps:
         1. Convert individual's bits to {+1, -1}
         2. Hopfield converge
         3. Compare stable state to the target pattern in {+1, -1}
        Fitness is the number of matching bits (max = rows*cols).
        """
        # Convert individual's bits to ±1
        init_pm1 = bits_to_pm1(individual)
        
        # Run Hopfield
        stable_pm1 = hopfield_converge(init_pm1, self.W, max_iter=20)
        
        # Compare stable state to target
        target_pm1 = bits_to_pm1(self.target_bits.flatten())  # shape (n,)
        matches = np.sum(stable_pm1 == target_pm1)
        
        return matches
    
    def select_parents(self, fitnesses):
        """
        Roulette-wheel selection
        """
        total_fit = sum(fitnesses)
        if total_fit <= 0:
            # All zero or negative => pick randomly
            idx1 = random.randrange(self.population_size)
            idx2 = random.randrange(self.population_size)
            return idx1, idx2
        
        # pick1
        pick1 = random.uniform(0, total_fit)
        running_sum = 0
        idx1 = 0
        for i, f in enumerate(fitnesses):
            running_sum += f
            if running_sum >= pick1:
                idx1 = i
                break
        
        # pick2
        pick2 = random.uniform(0, total_fit)
        running_sum = 0
        idx2 = 0
        for i, f in enumerate(fitnesses):
            running_sum += f
            if running_sum >= pick2:
                idx2 = i
                break
        
        return idx1, idx2
    
    def crossover(self, parent1, parent2):
        """
        One-point crossover in the flattened array.
        """
        cut = random.randint(1, self.n - 1)
        child1 = np.concatenate([parent1[:cut], parent2[cut:]])
        child2 = np.concatenate([parent2[:cut], parent1[cut:]])
        return child1, child2
    
    def mutate(self, individual):
        """
        Flip bits with probability = mutation_rate
        """
        for i in range(self.n):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual
    
    def run_one_generation(self):
        """
        Evaluate entire population, do selection, crossover, mutation.
        Track the best individual's fitness and genotype.
        """
        fitnesses = [self.evaluate_fitness(ind) for ind in self.population]
        
        # Track best
        best_idx = np.argmax(fitnesses)
        best_fit = fitnesses[best_idx]
        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            self.best_individual = self.population[best_idx].copy()
        
        # Reproduce
        new_population = []
        while len(new_population) < self.population_size:
            idx1, idx2 = self.select_parents(fitnesses)
            p1 = self.population[idx1]
            p2 = self.population[idx2]
            
            c1, c2 = self.crossover(p1, p2)
            c1 = self.mutate(c1)
            c2 = self.mutate(c2)
            
            new_population.append(c1)
            if len(new_population) < self.population_size:
                new_population.append(c2)
        
        self.population = new_population
    
    def get_best_individual(self):
        """
        Returns the best individual's bit array ( shape=(n,) ).
        If none found, return random.
        """
        if self.best_individual is None:
            return np.random.randint(0, 2, size=self.n)
        return self.best_individual
    
    def get_best_fitness(self):
        return self.best_fitness
    
    def get_target_bits(self):
        return self.target_bits
    
    def hopfield_converged_of_best(self):
        """
        Return the Hopfield stable 2D bit pattern of the best individual.
        That is, we take best_individual, run it through Hopfield, 
        and convert the stable pattern back to bits for display.
        """
        if self.best_individual is None:
            return np.zeros((self.rows, self.cols), dtype=int)
        
        init_pm1 = bits_to_pm1(self.best_individual)
        stable_pm1 = hopfield_converge(init_pm1, self.W)
        stable_bits = pm1_to_bits(stable_pm1)
        return stable_bits.reshape((self.rows, self.cols))


# ----------------------------------------------------------------------
#               PyQt Widgets to Display 2D Patterns
# ----------------------------------------------------------------------

class Pattern2DWidget(QWidget):
    """
    A QWidget that displays a 2D array of bits (0 or 1).
    1 => black square, 0 => white square
    """
    def __init__(self, rows=8, cols=8):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.grid_data = np.zeros((rows, cols), dtype=int)
        
        self.setMinimumSize(200, 200)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        
        margin = 10
        spacing = 1
        
        w_avail = self.width() - 2*margin
        h_avail = self.height() - 2*margin
        
        cell_w = w_avail // self.cols
        cell_h = h_avail // self.rows
        
        cell_size = min(cell_w, cell_h)
        
        for r in range(self.rows):
            for c in range(self.cols):
                x = margin + c * cell_size
                y = margin + r * cell_size
                
                if self.grid_data[r, c] == 1:
                    painter.setBrush(QColor("black"))
                else:
                    painter.setBrush(QColor("white"))
                painter.drawRect(x, y, cell_size - spacing, cell_size - spacing)
    
    def set_pattern(self, pattern_2d):
        """
        pattern_2d: shape (rows, cols) in {0,1}.
        """
        if pattern_2d.shape == (self.rows, self.cols):
            self.grid_data = pattern_2d
            self.update()


# ----------------------------------------------------------------------
#                  MainWindow - Putting it All Together
# ----------------------------------------------------------------------

from PyQt5.QtWidgets import QMessageBox

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("GA + Hopfield Convergence (Toy Example)")
        
        # GA parameters
        self.rows = 16
        self.cols = 16
        self.population_size = 20
        self.mutation_rate = 0.001
        
        # Create the GA
        self.ga = HopfieldGA(
            rows=self.rows, 
            cols=self.cols,
            population_size=self.population_size,
            mutation_rate=self.mutation_rate
        )
        
        # Create widgets
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Two Pattern2DWidget's: left = best individual's bits, right = converged Hopfield
        self.widget_initial = Pattern2DWidget(rows=self.rows, cols=self.cols)
        self.widget_converged = Pattern2DWidget(rows=self.rows, cols=self.cols)
        
        self.btn_next = QPushButton("Next Generation")
        self.btn_next.clicked.connect(self.handle_next_generation)
        
        self.btn_run = QPushButton("Run Continuously")
        self.btn_run.clicked.connect(self.handle_run_continuously)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.handle_stop)
        self.btn_stop.setEnabled(False)
        
        self.label_fitness = QLabel("Best Fitness: 0")
        
        # Layout
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.widget_initial)
        h_layout.addWidget(self.widget_converged)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_next)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)
        
        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addLayout(btn_layout)
        v_layout.addWidget(self.label_fitness)
        
        central_widget.setLayout(v_layout)
        
        # A timer for continuous evolution
        self.timer = QTimer()
        self.timer.timeout.connect(self.handle_next_generation)
        
        self.resize(600, 400)
        
        # Initialize display
        self.update_display()
    
    def handle_next_generation(self):
        """
        Run one generation, then update the display.
        """
        self.ga.run_one_generation()
        self.update_display()
    
    def handle_run_continuously(self):
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        # E.g., run every 300ms
        self.timer.start(300)
    
    def handle_stop(self):
        self.timer.stop()
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
    
    def update_display(self):
        """
        Update the two 2D widgets with:
          1) Best individual's raw bit pattern
          2) Its Hopfield-converged stable pattern
        Also show best fitness.
        """
        best_individual = self.ga.get_best_individual()
        best_individual_2d = best_individual.reshape((self.rows, self.cols))
        self.widget_initial.set_pattern(best_individual_2d)
        
        # Now show its converged Hopfield pattern
        converged = self.ga.hopfield_converged_of_best()
        self.widget_converged.set_pattern(converged)
        
        self.label_fitness.setText(
            f"Best Fitness: {self.ga.get_best_fitness()} / {self.rows*self.cols}"
        )


def main():
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

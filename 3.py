import sys
import random
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, 
    QHBoxLayout, QLabel, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QPainter, QColor

# ---------------------------
# Genetic Algorithm Class
# ---------------------------
class Simple2DGeneticAlgorithm:
    def __init__(self, rows=10, cols=10, population_size=20, mutation_rate=0.01):
        self.rows = rows
        self.cols = cols
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        # Create a random target pattern
        self.target = np.random.randint(0, 2, size=(rows, cols))
        
        # Initialize a population: a list of 2D arrays
        # We'll store each as a flattened 1D array of length = rows*cols
        self.population = [np.random.randint(0, 2, size=(rows * cols)) 
                           for _ in range(population_size)]
        
        # Keep track of the best individual
        self.best_individual = None
        self.best_fitness = -1
    
    def evaluate_fitness(self, individual):
        """
        Fitness = number of matching bits with the target pattern.
        """
        # individual is shape (rows*cols,)
        # target is shape (rows,cols), so flatten target
        target_flat = self.target.flatten()
        
        matches = np.sum(individual == target_flat)
        return matches  # the higher, the better
    
    def select_parents(self, fitnesses):
        """
        Stochastic selection of parents using 'roulette wheel' style:
        Probability of picking an individual ~ its fitness.
        """
        total_fit = sum(fitnesses)
        if total_fit == 0:
            # If all zero, pick randomly
            idx1 = random.randrange(self.population_size)
            idx2 = random.randrange(self.population_size)
            return idx1, idx2
        
        # Spin the wheel for first parent
        pick1 = random.uniform(0, total_fit)
        running_sum = 0
        idx1 = 0
        for i, f in enumerate(fitnesses):
            running_sum += f
            if running_sum >= pick1:
                idx1 = i
                break
        
        # Spin again for second parent
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
        1-point crossover: pick a random cut, combine slices of two parents.
        """
        length = self.rows * self.cols
        cut = random.randint(1, length - 1)
        child1 = np.concatenate([parent1[:cut], parent2[cut:]])
        child2 = np.concatenate([parent2[:cut], parent1[cut:]])
        return child1, child2
    
    def mutate(self, individual):
        """
        Flip bits with probability = self.mutation_rate.
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]  # Flip 0->1 or 1->0
        return individual
    
    def run_one_generation(self):
        """
        Perform one iteration of the GA: evaluate, select, crossover, mutate, replace population.
        Also track the best solution from the new generation.
        """
        # 1. Evaluate fitness
        fitnesses = [self.evaluate_fitness(ind) for ind in self.population]
        
        # Track best
        best_idx = np.argmax(fitnesses)
        best_fit = fitnesses[best_idx]
        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            self.best_individual = self.population[best_idx].copy()
        
        # 2. Create new population
        new_population = []
        while len(new_population) < self.population_size:
            # Select two parents
            idx1, idx2 = self.select_parents(fitnesses)
            parent1 = self.population[idx1]
            parent2 = self.population[idx2]
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutate
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Replace old population
        self.population = new_population
    
    def get_best_individual_2d(self):
        """
        Return the best individual's genotype as a 2D array (rows, cols).
        If we haven't found one yet, return a random array.
        """
        if self.best_individual is None:
            return np.random.randint(0, 2, size=(self.rows, self.cols))
        else:
            return self.best_individual.reshape((self.rows, self.cols))

    def get_best_fitness(self):
        return self.best_fitness
    
    def get_target_pattern(self):
        return self.target


# ---------------------------
# PyQt Widget to Display 2D Patterns
# ---------------------------
class GA2DWidget(QWidget):
    """
    Displays a 2D grid of bits. 1 => black, 0 => white.
    """
    def __init__(self, rows=10, cols=10):
        super().__init__()
        self.rows = rows
        self.cols = cols
        
        # This will store the 2D array of bits (0/1)
        self.grid_data = np.zeros((rows, cols), dtype=int)
        
        self.setMinimumSize(300, 300)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        
        margin = 10
        spacing = 2  # space between cells (optional)
        
        w_avail = self.width() - 2*margin
        h_avail = self.height() - 2*margin
        
        cell_size_w = w_avail // self.cols
        cell_size_h = h_avail // self.rows
        
        cell_size = min(cell_size_w, cell_size_h)
        
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
        Update the widget’s internal grid data and repaint.
        pattern_2d must be shape (rows, cols).
        """
        if pattern_2d.shape == (self.rows, self.cols):
            self.grid_data = pattern_2d
            self.update()


# ---------------------------
# Main Window with GA Integration
# ---------------------------
from PyQt5.QtWidgets import QMessageBox

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Genetic Algorithm (Toy Example)")
        
        # GA parameters
        self.rows = 10
        self.cols = 10
        self.population_size = 20
        self.mutation_rate = 0.02
        
        # Create GA instance
        self.ga = Simple2DGeneticAlgorithm(
            rows=self.rows, 
            cols=self.cols,
            population_size=self.population_size,
            mutation_rate=self.mutation_rate
        )
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.ga_widget = GA2DWidget(rows=self.rows, cols=self.cols)
        
        # Buttons
        self.btn_next_gen = QPushButton("Next Generation")
        self.btn_next_gen.clicked.connect(self.handle_next_generation)
        
        self.btn_run = QPushButton("Run Continuously")
        self.btn_run.clicked.connect(self.handle_run_continuously)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.handle_stop)
        self.btn_stop.setEnabled(False)
        
        # We might also display the best fitness so far
        self.label_fitness = QLabel("Best Fitness: 0")
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.ga_widget)
        
        h_btn_layout = QHBoxLayout()
        h_btn_layout.addWidget(self.btn_next_gen)
        h_btn_layout.addWidget(self.btn_run)
        h_btn_layout.addWidget(self.btn_stop)
        layout.addLayout(h_btn_layout)
        
        layout.addWidget(self.label_fitness)
        
        central_widget.setLayout(layout)
        
        # A timer to run continuous evolution
        self.timer = QTimer()
        self.timer.timeout.connect(self.handle_next_generation)
        
        self.resize(500, 550)
        
        # Initialize the widget display with the best individual's pattern
        self.update_ga_display()
    
    def handle_next_generation(self):
        """
        Run one GA generation, update display.
        """
        self.ga.run_one_generation()
        self.update_ga_display()
    
    def handle_run_continuously(self):
        """
        Start the timer to step the GA repeatedly.
        """
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        # For example, run at 200 ms interval
        self.timer.start(200)
    
    def handle_stop(self):
        """
        Stop continuous running.
        """
        self.timer.stop()
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
    
    def update_ga_display(self):
        """
        Get best individual's pattern from GA, display it, update label.
        """
        best = self.ga.get_best_individual_2d()
        self.ga_widget.set_pattern(best)
        
        best_fit = self.ga.get_best_fitness()
        # Max possible is rows*cols. Show progress fraction or raw value
        self.label_fitness.setText(f"Best Fitness: {best_fit} / {self.rows * self.cols}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

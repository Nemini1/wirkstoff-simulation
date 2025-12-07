"""
Protein-Ligand Simulation (Browser Optimized)
"""

import asyncio
import pygame
import random
import math
from dataclasses import dataclass
import sys
import os

# Imports
import pygame_gui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

# --- Constants ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SIDEBAR_WIDTH = 300

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 100, 150)
PROTEIN_COLOR = (255, 100, 100)
NORMAL_LIGAND_COLOR = (100, 150, 255)
COMPETITOR_LIGAND_COLOR = (255, 0, 0)
BOUND_LIGAND_COLOR = (255, 255, 0)
DOCKING_SITE_COLOR = (255, 255, 255)
DARK_GREY = (50, 50, 50)

@dataclass
class Parameters:
    num_proteins: int = 20
    num_ligands: int = 30
    num_competitor_ligands: int = 30
    temperature: float = 10.0
    k_on: float = 0.1
    k_on_competitor: float = 0.1
    k_off: float = 0.01
    binding_radius: float = 10.0
    dt: float = 10
    sim_area: tuple = (SCREEN_WIDTH - SIDEBAR_WIDTH, SCREEN_HEIGHT)

class Particle:
    def __init__(self, x, y, radius, color):
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(0, 0)
        self.radius = radius
        self.color = color

    def move_brownian(self, params):
        displacement_x = random.uniform(-1, 1)
        displacement_y = random.uniform(-1, 1)
        displacement_vector = pygame.Vector2(displacement_x, displacement_y)
        scale_factor = math.sqrt(params.temperature * params.dt)
        self.position += displacement_vector * scale_factor

    def check_wall_collision_and_bounce(self, sim_rect):
        if self.position.x - self.radius < sim_rect.left:
            self.position.x = sim_rect.left + self.radius
            self.velocity.x *= -1
        elif self.position.x + self.radius > sim_rect.right:
            self.position.x = sim_rect.right - self.radius
            self.velocity.x *= -1
        if self.position.y - self.radius < sim_rect.top:
            self.position.y = sim_rect.top + self.radius
            self.velocity.y *= -1
        elif self.position.y + self.radius > sim_rect.bottom:
            self.position.y = sim_rect.bottom - self.radius
            self.velocity.y *= -1

class Protein(Particle):
    radius = 15
    def __init__(self, x, y):
        super().__init__(x, y, radius=Protein.radius, color=PROTEIN_COLOR)
        self.bound_ligands = []

class Ligand(Particle):
    radius = 6
    def __init__(self, x, y):
        super().__init__(x, y, radius=Ligand.radius, color=NORMAL_LIGAND_COLOR)
        self.is_bound = False
        self.bound_to = None

    def unbind(self):
        if self.bound_to:
            try:
                self.bound_to.bound_ligands.remove(self)
            except ValueError:
                pass 
            
            direction_vector = self.position - self.bound_to.position
            separation_distance = self.bound_to.radius + self.radius + 2
            if direction_vector.length() > 0:
                direction_vector.normalize_ip()
            else:
                direction_vector = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
            self.position = self.bound_to.position + direction_vector * separation_distance
            kick_magnitude = random.uniform(1, 2)
            self.velocity = direction_vector * kick_magnitude
        self.is_bound = False
        self.bound_to = None
        self.color = self.get_initial_color()

    def get_initial_color(self):
        return NORMAL_LIGAND_COLOR

class CompetitorLigand(Ligand):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.color = COMPETITOR_LIGAND_COLOR
    
    def get_initial_color(self):
        return COMPETITOR_LIGAND_COLOR

class Simulation:
    def __init__(self):
        pygame.init()
        self.params = Parameters()
        self.paused = False
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Protein-Ligand Interaction Simulation")
        self.clock = pygame.time.Clock()
        self.sim_rect = pygame.Rect(0, 0, self.params.sim_area[0], self.params.sim_area[1])
        self.sidebar_rect = pygame.Rect(self.params.sim_area[0], 0, SIDEBAR_WIDTH, SCREEN_HEIGHT)
        self.font = pygame.font.Font(None, 24)
        
        self.proteins = []
        self.ligands = []
        self.competitor_ligands = []
        
        # --- UI SETUP ---
        # Wir überprüfen explizit, ob die Theme-Dateien da sind.
        if not os.path.exists("theme.json") or not os.path.exists("font.ttf"):
            print("FATAL ERROR: theme.json or font.ttf missing. JS loading failed.")
            # Wir beenden hier, weil pygame_gui sonst crasht
            return 

        try:
            self.ui_manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT), "theme.json")
        except Exception as e:
            print(f"UI Manager Init Error: {e}")
            raise e

        self._setup_ui_elements()
        self._initialize_particles()
        self._initialize_graph()

    def _setup_ui_elements(self):
        panel_rect = pygame.Rect(self.sidebar_rect.topleft, (self.sidebar_rect.width, self.sidebar_rect.height))
        self.ui_panel = pygame_gui.elements.UIPanel(relative_rect=panel_rect, manager=self.ui_manager)
        
        self.reset_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((20, 20), (self.sidebar_rect.width - 40, 40)), text='Reset Simulation', manager=self.ui_manager, container=self.ui_panel)
        self.pause_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((20, 70), (self.sidebar_rect.width / 2 - 25, 40)), text='Pause', manager=self.ui_manager, container=self.ui_panel)
        self.resume_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.sidebar_rect.width / 2 + 5, 70), (self.sidebar_rect.width / 2 - 25, 40)), text='Resume', manager=self.ui_manager, container=self.ui_panel)
        
        y_pos = 130; label_height = 30; slider_height = 20; slider_width = self.sidebar_rect.width - 40; x_pos = 20

        self.num_proteins_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, label_height)), text=f'Num Proteins: {self.params.num_proteins}', manager=self.ui_manager, container=self.ui_panel)
        y_pos += label_height
        self.num_proteins_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, slider_height)), start_value=self.params.num_proteins, value_range=(1, 150), manager=self.ui_manager, container=self.ui_panel)
        
        y_pos += slider_height + 20
        self.num_ligands_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, label_height)), text=f'Num Ligands: {self.params.num_ligands}', manager=self.ui_manager, container=self.ui_panel)
        y_pos += label_height
        self.num_ligands_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, slider_height)), start_value=self.params.num_ligands, value_range=(0, 300), manager=self.ui_manager, container=self.ui_panel)

        y_pos += slider_height + 20
        self.num_competitor_ligands_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, label_height)), text=f'Num Competitors: {self.params.num_competitor_ligands}', manager=self.ui_manager, container=self.ui_panel)
        y_pos += label_height
        self.num_competitor_ligands_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, slider_height)), start_value=self.params.num_competitor_ligands, value_range=(0, 300), manager=self.ui_manager, container=self.ui_panel)

        y_pos += slider_height + 20
        self.temp_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, label_height)), text=f'Temperature: {self.params.temperature:.2f}', manager=self.ui_manager, container=self.ui_panel)
        y_pos += label_height
        self.temp_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, slider_height)), start_value=self.params.temperature, value_range=(0, 100.0), manager=self.ui_manager, container=self.ui_panel)

        y_pos += slider_height + 20
        self.kon_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, label_height)), text=f'Ligand Binding Prob. (k_on): {self.params.k_on:.2f}', manager=self.ui_manager, container=self.ui_panel)
        y_pos += label_height
        self.kon_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, slider_height)), start_value=self.params.k_on, value_range=(0.0, 1.0), manager=self.ui_manager, container=self.ui_panel)

        y_pos += slider_height + 20
        self.kon_competitor_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, label_height)), text=f'Competitor Binding Prob. (k_on): {self.params.k_on_competitor:.2f}', manager=self.ui_manager, container=self.ui_panel)
        y_pos += label_height
        self.kon_competitor_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, slider_height)), start_value=self.params.k_on_competitor, value_range=(0.0, 1.0), manager=self.ui_manager, container=self.ui_panel)

    def _initialize_graph(self):
        self.time_steps = []
        self.bound_ligands_data = []
        self.bound_competitor_data = []

    def _update_graph_data(self):
        current_time_step = len(self.time_steps)
        self.time_steps.append(current_time_step)
        bound_ligands_count = 0
        bound_competitors_count = 0
        for protein in self.proteins:
            if protein.bound_ligands:
                bound_ligand = protein.bound_ligands[0]
                if isinstance(bound_ligand, CompetitorLigand):
                    bound_competitors_count += 1
                else:
                    bound_ligands_count += 1
        self.bound_ligands_data.append(bound_ligands_count)
        self.bound_competitor_data.append(bound_competitors_count)

    def _draw_graph(self):
        dark_grey_norm = tuple(c / 255.0 for c in DARK_GREY)
        white_norm = tuple(c / 255.0 for c in WHITE)
        
        fig = plt.Figure(figsize=(3, 2), dpi=100, facecolor=dark_grey_norm)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor(dark_grey_norm)
        ax.tick_params(colors=white_norm, labelsize=6)
        ax.spines['left'].set_color(white_norm)
        ax.spines['bottom'].set_color(white_norm)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        time_in_seconds = [t / 60.0 for t in self.time_steps]
        if len(time_in_seconds) > 500:
             time_in_seconds = time_in_seconds[-500:]
             data_ligands = self.bound_ligands_data[-500:]
             data_comp = self.bound_competitor_data[-500:]
        else:
             data_ligands = self.bound_ligands_data
             data_comp = self.bound_competitor_data

        ax.plot(time_in_seconds, data_ligands, color='#6496FF', label='Normal', linewidth=1)
        ax.plot(time_in_seconds, data_comp, color='red', label='Competitors', linewidth=1)
        
        ax.set_title("Bound Ligands", color=white_norm, fontsize=8)
        ax.set_ylim(0, self.params.num_proteins)
        ax.legend(loc='upper right', fontsize=5, frameon=False, labelcolor='white')
        
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.buffer_rgba()
        size = canvas.get_width_height()
        
        plt.close(fig)
        
        return pygame.image.frombuffer(raw_data, size, "RGBA")

    def _initialize_particles(self):
        self.proteins.clear()
        self.ligands.clear()
        self.competitor_ligands.clear()
        for _ in range(int(self.params.num_proteins)):
            protein = self._create_particle(Protein)
            self.proteins.append(protein)
        for _ in range(int(self.params.num_ligands)):
            ligand = self._create_particle(Ligand)
            self.ligands.append(ligand)
            ligand.unbind()
        for _ in range(int(self.params.num_competitor_ligands)):
            competitor = self._create_particle(CompetitorLigand)
            self.competitor_ligands.append(competitor)
            competitor.unbind()

    def _update_particle_counts(self):
        if len(self.proteins) < self.params.num_proteins:
            for _ in range(self.params.num_proteins - len(self.proteins)):
                self.proteins.append(self._create_particle(Protein))
        elif len(self.proteins) > self.params.num_proteins:
             for _ in range(len(self.proteins) - self.params.num_proteins):
                 if self.proteins: self.proteins.pop().bound_ligands.clear()
        if len(self.ligands) < self.params.num_ligands:
             for _ in range(self.params.num_ligands - len(self.ligands)):
                 l = self._create_particle(Ligand); self.ligands.append(l); l.unbind()
        elif len(self.ligands) > self.params.num_ligands:
             for _ in range(len(self.ligands) - self.params.num_ligands):
                 if self.ligands: self.ligands.pop()
        if len(self.competitor_ligands) < self.params.num_competitor_ligands:
             for _ in range(self.params.num_competitor_ligands - len(self.competitor_ligands)):
                 c = self._create_particle(CompetitorLigand); self.competitor_ligands.append(c); c.unbind()
        elif len(self.competitor_ligands) > self.params.num_competitor_ligands:
             for _ in range(len(self.competitor_ligands) - self.params.num_competitor_ligands):
                 if self.competitor_ligands: self.competitor_ligands.pop()

    def _create_particle(self, particle_class):
        all_particles = self.proteins + self.ligands + self.competitor_ligands
        for _ in range(100):
            new_x = random.randint(particle_class.radius, self.sim_rect.width - particle_class.radius)
            new_y = random.randint(particle_class.radius, self.sim_rect.height - particle_class.radius)
            new_particle = particle_class(new_x, new_y)
            overlap = False
            for particle in all_particles:
                dist = new_particle.position.distance_to(particle.position)
                if dist < new_particle.radius + particle.radius:
                    overlap = True; break
            if not overlap: return new_particle
        return particle_class(0,0)

    def _update_simulation(self):
        if not self.paused:
            all_movers = self.proteins + self.ligands + self.competitor_ligands
            for particle in all_movers:
                if isinstance(particle, Protein) or not particle.is_bound:
                    particle.move_brownian(self.params)
                    particle.check_wall_collision_and_bounce(self.sim_rect)
            for ligand in self.ligands + self.competitor_ligands:
                if ligand.is_bound:
                    if ligand.bound_to:
                        ligand.position = pygame.Vector2(ligand.bound_to.position)
                    
                    p_off = 1 - math.exp(-self.params.k_off * self.params.dt)
                    if random.random() < p_off: ligand.unbind()
                else:
                    for protein in self.proteins:
                        if not protein.bound_ligands:
                            dist = ligand.position.distance_to(protein.position)
                            if dist < protein.radius + self.params.binding_radius:
                                p_on = 1 - math.exp(-self.params.k_on_competitor * self.params.dt) if isinstance(ligand, CompetitorLigand) else 1 - math.exp(-self.params.k_on * self.params.dt)
                                if random.random() < p_on:
                                    ligand.is_bound = True; ligand.bound_to = protein
                                    ligand.color = BOUND_LIGAND_COLOR; protein.bound_ligands.append(ligand)
                                    break

    def _draw_ui(self):
        pygame.draw.rect(self.screen, BLUE, self.sim_rect)
        for protein in self.proteins:
            pygame.draw.circle(self.screen, protein.color, (int(protein.position.x), int(protein.position.y)), int(protein.radius))
            pygame.draw.circle(self.screen, DOCKING_SITE_COLOR, (int(protein.position.x), int(protein.position.y)), 5)
        for ligand in self.ligands + self.competitor_ligands:
            pygame.draw.circle(self.screen, ligand.color, (int(ligand.position.x), int(ligand.position.y)), int(ligand.radius))
        
        self.ui_manager.update(self.clock.get_time() / 1000.0)
        self.ui_manager.draw_ui(self.screen)
        
        graph_surf = self._draw_graph()
        self.screen.blit(graph_surf, (self.sidebar_rect.x + 20, 550))

    async def run(self):
        running = True
        while running:
            time_delta = self.clock.tick(60) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    elif event.key == pygame.K_SPACE: self.paused = not self.paused
                    elif event.key == pygame.K_r: self._initialize_particles(); self._initialize_graph()
                
                self.ui_manager.process_events(event)
                
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.reset_button: self._initialize_particles(); self._initialize_graph()
                    elif event.ui_element == self.pause_button: self.paused = True
                    elif event.ui_element == self.resume_button: self.paused = False
                if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                    if event.ui_element == self.num_proteins_slider:
                        self.params.num_proteins = int(event.value); self.num_proteins_label.set_text(f'Num Proteins: {self.params.num_proteins}'); self._update_particle_counts()
                    elif event.ui_element == self.num_ligands_slider:
                        self.params.num_ligands = int(event.value); self.num_ligands_label.set_text(f'Num Ligands: {self.params.num_ligands}'); self._update_particle_counts()
                    elif event.ui_element == self.num_competitor_ligands_slider:
                        self.params.num_competitor_ligands = int(event.value); self.num_competitor_ligands_label.set_text(f'Num Competitors: {self.params.num_competitor_ligands}'); self._update_particle_counts()
                    elif event.ui_element == self.temp_slider:
                        self.params.temperature = event.value; self.temp_label.set_text(f'Temperature: {event.value:.2f}')
                    elif event.ui_element == self.kon_slider:
                        self.params.k_on = event.value; self.kon_label.set_text(f'Ligand Binding Prob. (k_on): {event.value:.2f}')
                    elif event.ui_element == self.kon_competitor_slider:
                        self.params.k_on_competitor = event.value; self.kon_competitor_label.set_text(f'Competitor Binding Prob. (k_on): {event.value:.2f}')

            self._update_simulation()
            self._update_graph_data()
            self.screen.fill(BLACK)
            self._draw_ui()
            pygame.display.flip()
            
            await asyncio.sleep(0)

        pygame.quit()

async def main():
    simulation = Simulation()
    await simulation.run()

if __name__ == "__main__":
    asyncio.run(main())

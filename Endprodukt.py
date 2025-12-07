""" 

""" 

 

import sys 

import pygame 

import random 

import math 

from dataclasses import dataclass 

 

try: 

    import pygame_gui 

except ImportError: 

    print("Pygame_gui is not installed. Please install it with 'pip install pygame_gui'") 

    sys.exit() 

 

try: 

    import matplotlib.pyplot as plt 

    from matplotlib.backends.backend_agg import FigureCanvasAgg 

    import numpy as np 

except ImportError: 

    print("Matplotlib or Numpy is not installed. Please install it with 'pip install matplotlib numpy'") 

    sys.exit() 

 

# --- Constants and UI Configuration --- 

SCREEN_WIDTH = 1200 

SCREEN_HEIGHT = 800 

SIDEBAR_WIDTH = 300 

GRAPH_HEIGHT = 200 

 

# Colors (RGB) 

BLACK = (0, 0, 0) 

WHITE = (255, 255, 255) 

LIGHT_GREY = (200, 200, 200) 

DARK_GREY = (50, 50, 50) 

BLUE = (50, 100, 150) 

GREEN = (70, 130, 80) 

PROTEIN_COLOR = (255, 100, 100)  # Reddish 

NORMAL_LIGAND_COLOR = (100, 150, 255)  # Bluish 

COMPETITOR_LIGAND_COLOR = (255, 0, 0) # Red 

BOUND_LIGAND_COLOR = (255, 255, 0)  # Yellow 

DOCKING_SITE_COLOR = (255, 255, 255) # White 

 

@dataclass 

class Parameters: 

    """ 

    Centralized class to hold all simulation parameters. 

    All values are in normalized simulation units. 

    """ 

    num_proteins: int = 20 

    num_ligands: int = 30 

    num_competitor_ligands: int = 30 

    temperature: float = 10.0 

    k_on: float = 0.1 

    k_on_competitor: float = 0.1 

    k_off: float = 0.01 

    binding_radius: float = 10.0 

    dt: float = 10  # Timestep for simulation 

    sim_area: tuple = (SCREEN_WIDTH - SIDEBAR_WIDTH, SCREEN_HEIGHT) 

 

 

class Particle: 

    """ 

    Base class for any particle in the simulation (e.g., proteins, ligands). 

    """ 

    def __init__(self, x, y, radius, color): 

        self.position = pygame.Vector2(x, y) 

        self.velocity = pygame.Vector2(0, 0) 

        self.radius = radius 

        self.color = color 

 

    def move_brownian(self, params): 

        """ 

        Updates the particle's position with Brownian motion. 

        The displacement is a small random vector scaled by temperature and timestep. 

        """ 

        # Generate a random vector for displacement 

        displacement_x = random.uniform(-1, 1) 

        displacement_y = random.uniform(-1, 1) 

        displacement_vector = pygame.Vector2(displacement_x, displacement_y) 

 

        # Scale the displacement by the square root of the temperature and timestep 

        # This relationship is based on the Langevin equation for Brownian motion 

        scale_factor = math.sqrt(params.temperature * params.dt) 

        self.position += displacement_vector * scale_factor 

 

    def check_wall_collision_and_bounce(self, sim_rect): 

        """ 

        Checks if the particle has hit the boundaries of the simulation area 

        and reverses its velocity to simulate a bounce. 

        """ 

        # Check horizontal boundaries 

        if self.position.x - self.radius < sim_rect.left: 

            self.position.x = sim_rect.left + self.radius 

            self.velocity.x *= -1  # Reverse velocity 

        elif self.position.x + self.radius > sim_rect.right: 

            self.position.x = sim_rect.right - self.radius 

            self.velocity.x *= -1 

 

        # Check vertical boundaries 

        if self.position.y - self.radius < sim_rect.top: 

            self.position.y = sim_rect.top + self.radius 

            self.velocity.y *= -1 

        elif self.position.y + self.radius > sim_rect.bottom: 

            self.position.y = sim_rect.bottom - self.radius 

            self.velocity.y *= -1 

 

class Protein(Particle): 

    """A Protein particle, larger in size, with a binding site.""" 

    radius = 15 

    def __init__(self, x, y): 

        super().__init__(x, y, radius=Protein.radius, color=PROTEIN_COLOR) 

        self.bound_ligands = [] 

 

class Ligand(Particle): 

    """A Ligand particle, smaller in size, that can bind to proteins.""" 

    radius = 6 

    def __init__(self, x, y): 

        super().__init__(x, y, radius=Ligand.radius, color=NORMAL_LIGAND_COLOR) 

        self.is_bound = False 

        self.bound_to = None 

 

    def unbind(self): 

        """ 

        Handles the unbinding logic, including resetting the ligand's state 

        and removing it from the protein's list of bound ligands. 

        Also repositions the ligand to "pop" it off the protein. 

        """ 

        if self.bound_to: 

            # Safely remove the ligand from the protein's list 

            self.bound_to.bound_ligands.remove(self) 

         

            # Calculate the direction vector from the protein to the ligand 

            direction_vector = self.position - self.bound_to.position 

             

            # Reposition the ligand just outside the protein's radius 

            # Use a small buffer (e.g., +2) to prevent immediate re-binding 

            separation_distance = self.bound_to.radius + self.radius + 2 

             

            # Normalize the direction vector to get a unit vector 

            if direction_vector.length() > 0: 

                direction_vector.normalize_ip() 

            else: 

                # Handle case where ligand is exactly at protein's center 

                direction_vector = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() 

 

            # Set the new position 

            self.position = self.bound_to.position + direction_vector * separation_distance 

 

            # Give it a small outward "kick" velocity 

            kick_magnitude = random.uniform(1, 2) 

            self.velocity = direction_vector * kick_magnitude 

 

        # Reset the ligand's binding state 

        self.is_bound = False 

        self.bound_to = None 

        self.color = self.get_initial_color() 

 

    def get_initial_color(self): 

        return NORMAL_LIGAND_COLOR 

 

class CompetitorLigand(Ligand): 

    """A Ligand particle that competes with normal ligands for binding sites.""" 

    def __init__(self, x, y): 

        super().__init__(x, y) 

        self.color = COMPETITOR_LIGAND_COLOR 

     

    def get_initial_color(self): 

        return COMPETITOR_LIGAND_COLOR 

 

class Simulation: 

    """ 

    Main class to handle the simulation logic and Pygame rendering. 

    """ 

 

    def __init__(self): 

        """ 

        Initializes the Pygame environment, parameters, and UI elements. 

        """ 

        pygame.init() 

        self.params = Parameters() 

        self.paused = False 

 

        # Pygame window and clock setup 

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) 

        pygame.display.set_caption("Protein-Ligand Interaction Simulation") 

        self.clock = pygame.time.Clock() 

        self.time_steps = [] 

        self.bound_ligands_data = [] 

        self.bound_competitor_data = [] 

 

        # Define the areas for the main canvas and the sidebar 

        self.sim_rect = pygame.Rect(0, 0, self.params.sim_area[0], self.params.sim_area[1]) 

        self.sidebar_rect = pygame.Rect(self.params.sim_area[0], 0, SIDEBAR_WIDTH, SCREEN_HEIGHT) 

 

        # Font for text rendering 

        self.font = pygame.font.Font(None, 24) 

        self.title_font = pygame.font.Font(None, 32) 

         

        # Lists to hold all simulation particles 

        self.proteins = [] 

        self.ligands = [] 

        self.competitor_ligands = [] 

 

        # Setup GUI elements 

        self.ui_manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT)) 

        self._setup_ui_elements() 

         

        # Initialize particles and graph data 

        self._initialize_particles() 

        self._initialize_graph() 

     

    def _setup_ui_elements(self): 

        """ 

        Creates the UI sliders and labels in the sidebar using pygame_gui. 

        """ 

        # Create a UI panel to contain the sliders 

        panel_rect = pygame.Rect(self.sidebar_rect.topleft, (self.sidebar_rect.width, self.sidebar_rect.height)) 

        self.ui_panel = pygame_gui.elements.UIPanel(relative_rect=panel_rect, 

                                                     manager=self.ui_manager) 

         

        # Reset Button 

        self.reset_button = pygame_gui.elements.UIButton( 

            relative_rect=pygame.Rect((20, 20), (self.sidebar_rect.width - 40, 40)), 

            text='Reset Simulation', 

            manager=self.ui_manager, 

            container=self.ui_panel) 

         

        # Pause/Resume Buttons 

        self.pause_button = pygame_gui.elements.UIButton( 

            relative_rect=pygame.Rect((20, 70), (self.sidebar_rect.width / 2 - 25, 40)), 

            text='Pause', 

            manager=self.ui_manager, 

            container=self.ui_panel) 

 

        self.resume_button = pygame_gui.elements.UIButton( 

            relative_rect=pygame.Rect((self.sidebar_rect.width / 2 + 5, 70), (self.sidebar_rect.width / 2 - 25, 40)), 

            text='Resume', 

            manager=self.ui_manager, 

            container=self.ui_panel) 

 

        # Labels and sliders for parameters 

        y_pos = 130 

        label_height = 30 

        slider_height = 20 

        slider_width = self.sidebar_rect.width - 40 

        x_pos = 20 

 

        # Number of Proteins Slider 

        self.num_proteins_label = pygame_gui.elements.UILabel( 

            relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, label_height)), 

            text=f'Num Proteins: {self.params.num_proteins}', 

            manager=self.ui_manager, 

            container=self.ui_panel) 

        y_pos += label_height 

        self.num_proteins_slider = pygame_gui.elements.UIHorizontalSlider( 

            relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, slider_height)), 

            start_value=self.params.num_proteins, 

            value_range=(1, 150), 

            manager=self.ui_manager, 

            container=self.ui_panel) 

         

        # Number of Ligands Slider 

        y_pos += slider_height + 20 

        self.num_ligands_label = pygame_gui.elements.UILabel( 

            relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, label_height)), 

            text=f'Num Ligands: {self.params.num_ligands}', 

            manager=self.ui_manager, 

            container=self.ui_panel) 

        y_pos += label_height 

        self.num_ligands_slider = pygame_gui.elements.UIHorizontalSlider( 

            relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, slider_height)), 

            start_value=self.params.num_ligands, 

            value_range=(0, 300), 

            manager=self.ui_manager, 

            container=self.ui_panel) 

         

        # Number of Competitor Ligands Slider 

        y_pos += slider_height + 20 

        self.num_competitor_ligands_label = pygame_gui.elements.UILabel( 

            relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, label_height)), 

            text=f'Num Competitors: {self.params.num_competitor_ligands}', 

            manager=self.ui_manager, 

            container=self.ui_panel) 

        y_pos += label_height 

        self.num_competitor_ligands_slider = pygame_gui.elements.UIHorizontalSlider( 

            relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, slider_height)), 

            start_value=self.params.num_competitor_ligands, 

            value_range=(0, 300), 

            manager=self.ui_manager, 

            container=self.ui_panel) 

 

        # Temperature Slider 

        y_pos += slider_height + 20 

        self.temp_label = pygame_gui.elements.UILabel( 

            relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, label_height)), 

            text=f'Temperature: {self.params.temperature:.2f}', 

            manager=self.ui_manager, 

            container=self.ui_panel) 

        y_pos += label_height 

        self.temp_slider = pygame_gui.elements.UIHorizontalSlider( 

            relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, slider_height)), 

            start_value=self.params.temperature, 

            value_range=(0, 100.0), 

            manager=self.ui_manager, 

            container=self.ui_panel) 

         

        # Binding Probability (k_on) Slider 

        y_pos += slider_height + 20 

        self.kon_label = pygame_gui.elements.UILabel( 

            relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, label_height)), 

            text=f'Ligand Binding Prob. (k_on): {self.params.k_on:.2f}', 

            manager=self.ui_manager, 

            container=self.ui_panel) 

        y_pos += label_height 

        self.kon_slider = pygame_gui.elements.UIHorizontalSlider( 

            relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, slider_height)), 

            start_value=self.params.k_on, 

            value_range=(0.0, 1.0), 

            manager=self.ui_manager, 

            container=self.ui_panel) 

 

        # Competitor Binding Probability (k_on_competitor) Slider 

        y_pos += slider_height + 20 

        self.kon_competitor_label = pygame_gui.elements.UILabel( 

            relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, label_height)), 

            text=f'Competitor Binding Prob. (k_on): {self.params.k_on_competitor:.2f}', 

            manager=self.ui_manager, 

            container=self.ui_panel) 

        y_pos += label_height 

        self.kon_competitor_slider = pygame_gui.elements.UIHorizontalSlider( 

            relative_rect=pygame.Rect((x_pos, y_pos), (slider_width, slider_height)), 

            start_value=self.params.k_on_competitor, 

            value_range=(0.0, 1.0), 

            manager=self.ui_manager, 

            container=self.ui_panel) 

 

 

    def _initialize_graph(self): 

        """ 

        Initializes the matplotlib figure for the real-time graph. 

        """ 

        self.time_steps = [] 

        self.bound_ligands_data = [] 

        self.bound_competitor_data = [] 

 

    def _update_graph_data(self): 

        """ 

        Appends the current simulation data to the graph lists. 

        """ 

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

        """ 

        Draws the real-time graph and returns it as a Pygame surface. 

        """ 

        # Normalize the colors from 0-255 to 0-1 for Matplotlib 

        dark_grey_norm = tuple(c / 255.0 for c in DARK_GREY) 

        white_norm = tuple(c / 255.0 for c in WHITE) 

 

        fig = plt.Figure(figsize=(3, 2), dpi=100, facecolor=dark_grey_norm) 

        canvas = FigureCanvasAgg(fig) 

        ax = fig.add_subplot(111) 

        ax.set_facecolor(dark_grey_norm) 

        ax.tick_params(colors=white_norm) 

        ax.spines['left'].set_color(white_norm) 

        ax.spines['bottom'].set_color(white_norm) 

 

        # Convert ticks to seconds for the x-axis 

        time_in_seconds = [t / 60.0 for t in self.time_steps] 

         

        # Plot both curves 

        ax.plot(time_in_seconds, self.bound_ligands_data, color='blue', label='Normal Ligands') 

        ax.plot(time_in_seconds, self.bound_competitor_data, color='red', label='Competitors') 

         

        ax.set_title("Bound Ligands Over Time", color=white_norm, fontsize=8) 

        ax.set_xlabel("Time (s)", color=white_norm, fontsize=6) 

        ax.set_ylabel("Count", color=white_norm, fontsize=6) 

         

        ax.set_ylim(0, self.params.num_proteins) 

        ax.legend(loc='upper right', fontsize=6, frameon=False, labelcolor=white_norm) 

         

        canvas.draw() 

        renderer = canvas.get_renderer() 

        raw_data = renderer.buffer_rgba() 

        size = canvas.get_width_height() 

 

        surf = pygame.image.frombuffer(raw_data, size, "RGBA") 

        return surf 

 

    def _initialize_particles(self): 

        """ 

        Creates and positions the initial set of proteins and ligands based on current parameters. 

        Ensures they do not overlap at the start. 

        """ 

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

        """ 

        Dynamically adds or removes proteins and ligands based on slider values. 

        """ 

        # Update protein count 

        num_current_proteins = len(self.proteins) 

        num_desired_proteins = int(self.params.num_proteins) 

         

        if num_desired_proteins > num_current_proteins: 

            for _ in range(num_desired_proteins - num_current_proteins): 

                new_protein = self._create_particle(Protein) 

                self.proteins.append(new_protein) 

        elif num_desired_proteins < num_current_proteins: 

            for _ in range(num_current_proteins - num_desired_proteins): 

                if self.proteins: 

                    removed_protein = self.proteins.pop() 

                    # Unbind any ligands bound to the removed protein 

                    for ligand in removed_protein.bound_ligands[:]: 

                        ligand.unbind() 

 

        # Update normal ligand count 

        num_current_ligands = len(self.ligands) 

        num_desired_ligands = int(self.params.num_ligands) 

         

        if num_desired_ligands > num_current_ligands: 

            for _ in range(num_desired_ligands - num_current_ligands): 

                new_ligand = self._create_particle(Ligand) 

                self.ligands.append(new_ligand) 

                new_ligand.unbind() 

        elif num_desired_ligands < num_current_ligands: 

            for _ in range(num_current_ligands - num_desired_ligands): 

                if self.ligands: 

                    self.ligands[-1].unbind() 

                    self.ligands.pop() 

 

        # Update competitor ligand count 

        num_current_competitors = len(self.competitor_ligands) 

        num_desired_competitors = int(self.params.num_competitor_ligands) 

         

        if num_desired_competitors > num_current_competitors: 

            for _ in range(num_desired_competitors - num_current_competitors): 

                new_competitor = self._create_particle(CompetitorLigand) 

                self.competitor_ligands.append(new_competitor) 

                new_competitor.unbind() 

        elif num_desired_competitors < num_current_competitors: 

            for _ in range(num_current_competitors - num_desired_competitors): 

                if self.competitor_ligands: 

                    self.competitor_ligands[-1].unbind() 

                    self.competitor_ligands.pop() 

 

 

    def _create_particle(self, particle_class): 

        """ 

        Creates a new particle instance at a random, non-overlapping position. 

        """ 

        all_particles = self.proteins + self.ligands + self.competitor_ligands 

        while True: 

            # Generate a random position within the simulation area 

            new_x = random.randint(particle_class.radius, self.sim_rect.width - particle_class.radius) 

            new_y = random.randint(particle_class.radius, self.sim_rect.height - particle_class.radius) 

            new_particle = particle_class(new_x, new_y) 

 

            # Check for overlap with existing particles 

            overlap = False 

            for particle in all_particles: 

                dist = new_particle.position.distance_to(particle.position) 

                if dist < new_particle.radius + particle.radius: 

                    overlap = True 

                    break 

             

            if not overlap: 

                return new_particle 

 

    def _update_simulation(self): 

        """ 

        Updates the state of the simulation for one timestep. 

        """ 

        if not self.paused: 

            # Update all particles 

            all_movers = self.proteins + self.ligands + self.competitor_ligands 

            for particle in all_movers: 

                # Update position for free particles 

                if isinstance(particle, Protein) or not particle.is_bound: 

                    particle.move_brownian(self.params) 

                    particle.check_wall_collision_and_bounce(self.sim_rect) 

 

            # Check for binding/unbinding for both ligand types 

            for ligand in self.ligands + self.competitor_ligands: 

                # If bound, move with the protein and check for unbinding 

                if ligand.is_bound: 

                    ligand.position = ligand.bound_to.position 

                     

                    # Unbinding probability 

                    p_off = 1 - math.exp(-self.params.k_off * self.params.dt) 

                    if random.random() < p_off: 

                        ligand.unbind() 

                # If free, check for binding with each protein 

                else: 

                    for protein in self.proteins: 

                        # Binding is only possible if the protein has an open binding site 

                        if not protein.bound_ligands: 

                            dist = ligand.position.distance_to(protein.position) 

                            if dist < protein.radius + self.params.binding_radius: 

                                # Determine binding probability based on ligand type 

                                p_on = 0 

                                if isinstance(ligand, CompetitorLigand): 

                                    p_on = 1 - math.exp(-self.params.k_on_competitor * self.params.dt) 

                                else: 

                                    p_on = 1 - math.exp(-self.params.k_on * self.params.dt) 

 

                                if random.random() < p_on: 

                                    ligand.is_bound = True 

                                    ligand.bound_to = protein 

                                    ligand.color = BOUND_LIGAND_COLOR 

                                    protein.bound_ligands.append(ligand) 

                                    break # Ligand can only bind to one protein at a time 

 

    def _draw_ui(self): 

        """ 

        Draws the user interface, including the main canvas and the sidebar. 

        """ 

        # Draw the main simulation canvas 

        pygame.draw.rect(self.screen, BLUE, self.sim_rect) 

 

        # Draw the protein and its binding site 

        for protein in self.proteins: 

            pygame.draw.circle(self.screen, protein.color, (int(protein.position.x), int(protein.position.y)), int(protein.radius)) 

            # Draw the docking site 

            pygame.draw.circle(self.screen, DOCKING_SITE_COLOR, (int(protein.position.x), int(protein.position.y)), 5) 

         

        # Draw the ligand 

        for ligand in self.ligands + self.competitor_ligands: 

            pygame.draw.circle(self.screen, ligand.color, (int(ligand.position.x), int(ligand.position.y)), int(ligand.radius)) 

 

        # Update and draw the GUI elements 

        self.ui_manager.update(self.clock.get_time() / 1000.0) 

        self.ui_manager.draw_ui(self.screen) 

         

        # Draw the graph 

        graph_surf = self._draw_graph() 

        self.screen.blit(graph_surf, (self.sidebar_rect.x + 20, 550)) 

 

 

    def run(self): 

        """ 

        The main simulation loop. Handles events and updates the display. 

        """ 

        running = True 

        while running: 

            time_delta = self.clock.tick(60) / 1000.0 

            # --- Event Handling --- 

            for event in pygame.event.get(): 

                if event.type == pygame.QUIT: 

                    running = False 

                elif event.type == pygame.KEYDOWN: 

                    if event.key == pygame.K_ESCAPE: 

                        running = False 

                    elif event.key == pygame.K_SPACE: 

                        self.paused = not self.paused 

                    elif event.key == pygame.K_r: 

                        self._initialize_particles() 

                        self._initialize_graph() 

                 

                # Handle GUI events 

                self.ui_manager.process_events(event) 

                 

                # Check for button clicks 

                if event.type == pygame_gui.UI_BUTTON_PRESSED: 

                    if event.ui_element == self.reset_button: 

                        self._initialize_particles() 

                        self._initialize_graph() 

                    elif event.ui_element == self.pause_button: 

                        self.paused = True 

                    elif event.ui_element == self.resume_button: 

                        self.paused = False 

                 

                # Check for slider updates 

                if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED: 

                    if event.ui_element == self.num_proteins_slider: 

                        new_value = int(event.value) 

                        self.params.num_proteins = new_value 

                        self.num_proteins_label.set_text(f'Num Proteins: {new_value}') 

                        self._update_particle_counts() 

                    elif event.ui_element == self.num_ligands_slider: 

                        new_value = int(event.value) 

                        self.params.num_ligands = new_value 

                        self.num_ligands_label.set_text(f'Num Ligands: {new_value}') 

                        self._update_particle_counts() 

                    elif event.ui_element == self.num_competitor_ligands_slider: 

                        new_value = int(event.value) 

                        self.params.num_competitor_ligands = new_value 

                        self.num_competitor_ligands_label.set_text(f'Num Competitors: {new_value}') 

                        self._update_particle_counts() 

                    elif event.ui_element == self.temp_slider: 

                        new_value = event.value 

                        self.params.temperature = new_value 

                        self.temp_label.set_text(f'Temperature: {new_value:.2f}') 

                    elif event.ui_element == self.kon_slider: 

                        new_value = event.value 

                        self.params.k_on = new_value 

                        self.kon_label.set_text(f'Ligand Binding Prob. (k_on): {new_value:.2f}') 

                    elif event.ui_element == self.kon_competitor_slider: 

                        new_value = event.value 

                        self.params.k_on_competitor = new_value 

                        self.kon_competitor_label.set_text(f'Competitor Binding Prob. (k_on): {new_value:.2f}') 

                 

            # --- Update Simulation State --- 

            self._update_simulation() 

            self._update_graph_data() 

             

            # --- Drawing --- 

            self.screen.fill(BLACK)  # Fill the background 

 

            # Draw the UI elements 

            self._draw_ui() 

 

            # --- Update Display --- 

            pygame.display.flip() 

 

        pygame.quit() 

        sys.exit() 

 

 

def main(): 

    """ 

    Main function to initialize and run the simulation. 

    """ 

    simulation = Simulation() 

    simulation.run() 

 

 

if __name__ == "__main__": 

    main() 

 

 
"""
Enhanced Flappy Bird with NEAT Algorithm
Includes real-time neural network visualization and fitness graph
"""
import pygame
import random
import os
import neat
import pickle
import visualize
pygame.font.init()

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
STAT_FONT = pygame.font.SysFont("", 30)
SMALL_FONT = pygame.font.SysFont("", 20)
END_FONT = pygame.font.SysFont("", 50)
DRAW_LINES = True

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird - NEAT AI with Visualizations")

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())

gen = 0
fitness_history = []
best_fitness_history = []
avg_fitness_history = []

class Bird:
    MAX_ROTATION = 20
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        displacement = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2

        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe():
    GAP = 184
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)

        if b_point or t_point:
            return True
        return False


class Base:
    VEL = 6
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect.topleft)


def draw_neural_network(win, genome, config, bird, pipes, pipe_ind, x_offset=50, y_offset=490):
    """Draw a simplified neural network visualization with live activations"""
    s = pygame.Surface((320, 330), pygame.SRCALPHA)
    s.fill((0, 0, 0, 100))
    win.blit(s, (x_offset-50, y_offset))
    
    # Border
    pygame.draw.rect(win, (150, 150, 150), (x_offset-50, y_offset, 260+60, 330), 2)
    
    title = SMALL_FONT.render("Best Neural Network", 1, (255, 255, 255))
    win.blit(title, (x_offset + 10, y_offset + 5))
    
    input_y_positions = [y_offset + 60, y_offset + 120, y_offset + 180]
    output_y_position = y_offset + 120
    
    input_x = x_offset + 50
    output_x = x_offset + 210
    
    # Safe pipe index checking
    if bird and pipes and len(pipes) > 0:
        # Make sure pipe_ind is valid
        if pipe_ind >= len(pipes):
            pipe_ind = 0
        
        try:
            inputs = [bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)]
            input_activations = [min(1.0, max(0.0, val / 800)) for val in inputs]
        except (IndexError, AttributeError):
            input_activations = [0, 0, 0]
    else:
        input_activations = [0, 0, 0]
    
    output_activation = 0
    connection_count = 0
    for conn_key, conn in genome.connections.items():
        if conn.enabled:
            in_node, out_node = conn_key
            
            if in_node in config.genome_config.input_keys:
                idx = config.genome_config.input_keys.index(in_node)
                start_pos = (input_x, input_y_positions[idx])
                activation = input_activations[idx]
            else:
                continue
            
            end_pos = (output_x, output_y_position)
            
            weight = conn.weight
            intensity = int(min(255, abs(weight) * 100 + 50))
            
            if weight > 0:
                color = (0, intensity, 255) 
            else:
                color = (intensity, 0, 255) 
            
            thickness = max(2, min(5, int(abs(weight) * 3)))
            pygame.draw.line(win, color, start_pos, end_pos, thickness)
            
            output_activation += weight * activation
            connection_count += 1
    
    if connection_count == 0:
        no_conn = SMALL_FONT.render("Evolving...", 1, (150, 150, 150))
        win.blit(no_conn, (x_offset + 80, output_y_position))
    
    output_activation = max(0, min(1, (output_activation + 1) / 2))
    
    input_labels = ["Bird Y", "Top Pipe", "Bottom Pipe"]
    for i, label in enumerate(input_labels):
        activation = input_activations[i]
        glow_intensity = int(min(155, activation * 200))

        for radius in range(20, 15, -1):
            alpha = min(255, int((20 - radius) * 15 * activation))
            s_glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s_glow, (255, 255, 100, alpha), (radius, radius), radius)
            win.blit(s_glow, (input_x - radius, input_y_positions[i] - radius))
        
        node_color = (min(255, 100 + glow_intensity), min(255, 100 + glow_intensity), 255)
        pygame.draw.circle(win, node_color, (input_x, input_y_positions[i]), 15)
        pygame.draw.circle(win, (255, 255, 255), (input_x, input_y_positions[i]), 15, 2)
        
        text = SMALL_FONT.render(label, 1, (255, 255, 255))
        win.blit(text, (input_x - 95, input_y_positions[i] - 10))
    
    glow_intensity = int(min(155, output_activation * 200))
    
    for radius in range(20, 15, -1):
        alpha = min(255, int((20 - radius) * 15 * output_activation))
        s_glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s_glow, (255, 100, 100, alpha), (radius, radius), radius)
        win.blit(s_glow, (output_x - radius, output_y_position - radius))
    
    node_color = (255, min(255, 100 + glow_intensity // 2), min(255, 100 + glow_intensity // 2))
    pygame.draw.circle(win, node_color, (output_x, output_y_position), 15)
    pygame.draw.circle(win, (255, 255, 255), (output_x, output_y_position), 15, 2)
    
    text = SMALL_FONT.render("Jump", 1, (255, 255, 255))
    win.blit(text, (output_x + 20, output_y_position - 10))
    
    info_y = y_offset + 220
    nodes_text = SMALL_FONT.render(f"Nodes: {len(genome.nodes)}", 1, (255, 255, 255))
    win.blit(nodes_text, (x_offset - 0, info_y))
    
    connections_text = SMALL_FONT.render(f"Connections: {len(genome.connections)}", 1, (255, 255, 255))
    win.blit(connections_text, (x_offset - 0, info_y + 25))
    
    fitness_text = SMALL_FONT.render(f"Fitness: {genome.fitness:.1f}", 1, (255, 255, 255))
    win.blit(fitness_text, (x_offset - 0, info_y + 50))

def draw_window(win, birds, pipes, base, score, gen, pipe_ind, best_genome, best_bird, config):
    if gen == 0:
        gen = 1
    win.blit(bg_img, (0,0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    
    for bird in birds:
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 1)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 1)
            except:
                pass
        bird.draw(win)

    score_label = STAT_FONT.render("Score: " + str(score),1,(255,255,255))
    win.blit(score_label, (10, 90))

    gen_label = STAT_FONT.render("Generation: " + str(gen-1),1,(255,255,255))
    win.blit(gen_label, (10, 10))

    alive_label = STAT_FONT.render("Alive: " + str(len(birds)) + " / 30 x 2 (batch)",1,(255,255,255))
    win.blit(alive_label, (10, 50))

    # draw_fitness_graph(win)
    if best_genome and best_bird and pipes:
        draw_neural_network(win, best_genome, config, best_bird, pipes, pipe_ind)

    pygame.display.update()


def eval_genomes(genomes, config):
    global WIN, gen, best_fitness_history, avg_fitness_history
    win = WIN
    gen += 1

    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230,350))
        ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()
    best_genome = ge[0]
    best_bird = birds[0]

    run = True
    while run and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        if ge:
            best_idx = ge.index(max(ge, key=lambda g: g.fitness))
            best_genome = ge[best_idx]
            best_bird = birds[best_idx]

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1
            bird.move()

            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            for bird in birds:
                if pipe.collide(bird, win):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        draw_window(WIN, birds, pipes, base, score, gen, pipe_ind, best_genome, best_bird, config)

    if ge:
        best_fitness = max([g.fitness for g in ge])
        avg_fitness = sum([g.fitness for g in ge]) / len(ge)
    else:
        best_fitness = 0
        avg_fitness = 0
    
    best_fitness_history.append(best_fitness)
    avg_fitness_history.append(avg_fitness)


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)

    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
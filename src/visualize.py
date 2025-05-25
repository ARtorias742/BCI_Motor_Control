# src/visualize.py
import pygame

class CursorVisualizer:
    def __init__(self, config):
        self.screen_width = config['visualization']['screen_width']
        self.screen_height = config['visualization']['screen_height']
        self.cursor_speed = config['visualization']['cursor_speed']
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.cursor_pos = self.screen_width // 2
        self.cursor = pygame.Rect(self.cursor_pos, self.screen_height // 2, 20, 20)

    def update(self, command):
        """Update cursor position based on command (0: left, 1: right)."""
        if command == 0:  # Move left
            self.cursor_pos -= self.cursor_speed
        elif command == 1:  # Move right
            self.cursor_pos += self.cursor_speed

        # Keep cursor within screen bounds
        self.cursor_pos = max(0, min(self.cursor_pos, self.screen_width - 20))
        self.cursor.x = self.cursor_pos

        # Draw
        self.screen.fill((0, 0, 0))  # Black background
        pygame.draw.rect(self.screen, (255, 255, 255), self.cursor)  # White cursor
        pygame.display.flip()

    def quit(self):
        pygame.quit()
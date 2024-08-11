import cv2
import numpy as np
import os

def create_synthetic_dataset(output_dir, num_samples=1000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'irregular'))
        os.makedirs(os.path.join(output_dir, 'regular'))

    for i in range(num_samples):
        # Create a blank image
        image = np.zeros((128, 128), dtype=np.uint8)

        # Randomly choose a shape
        shape_type = np.random.choice(['circle', 'square', 'triangle', 'ellipse'])
        irregular_shape, regular_shape = None, None

        if shape_type == 'circle':
            # Create a circle with noise
            center = (np.random.randint(40, 88), np.random.randint(40, 88))
            radius = np.random.randint(20, 40)
            irregular_shape = cv2.circle(image.copy(), center, radius + np.random.randint(-5, 5), 255, -1)
            regular_shape = cv2.circle(image.copy(), center, radius, 255, -1)

        elif shape_type == 'square':
            # Create a square with noise
            side = np.random.randint(40, 60)
            top_left = (np.random.randint(20, 68), np.random.randint(20, 68))
            irregular_shape = cv2.rectangle(image.copy(), top_left, (top_left[0] + side + np.random.randint(-10, 10), top_left[1] + side + np.random.randint(-10, 10)), 255, -1)
            regular_shape = cv2.rectangle(image.copy(), top_left, (top_left[0] + side, top_left[1] + side), 255, -1)

        elif shape_type == 'triangle':
            # Create a triangle with noise
            points = np.array([
                [np.random.randint(20, 108), np.random.randint(20, 40)],
                [np.random.randint(20, 40), np.random.randint(80, 108)],
                [np.random.randint(80, 108), np.random.randint(80, 108)]
            ])
            irregular_shape = cv2.drawContours(image.copy(), [points + np.random.randint(-5, 5, size=points.shape)], 0, 255, -1)
            regular_shape = cv2.drawContours(image.copy(), [points], 0, 255, -1)

        elif shape_type == 'ellipse':
            # Create an ellipse with noise
            center = (np.random.randint(40, 88), np.random.randint(40, 88))
            axes = (np.random.randint(20, 40), np.random.randint(10, 30))
            angle = np.random.randint(0, 360)
            irregular_shape = cv2.ellipse(image.copy(), center, (axes[0] + np.random.randint(-5, 5), axes[1] + np.random.randint(-5, 5)), angle, 0, 360, 255, -1)
            regular_shape = cv2.ellipse(image.copy(), center, axes, angle, 0, 360, 255, -1)

        # Save the pair
        cv2.imwrite(os.path.join(output_dir, 'irregular', f'irregular_{i}.png'), irregular_shape)
        cv2.imwrite(os.path.join(output_dir, 'regular', f'regular_{i}.png'), regular_shape)

# Generate the dataset
create_synthetic_dataset('synthetic_shapes', num_samples=1000)

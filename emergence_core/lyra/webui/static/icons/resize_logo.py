from PIL import Image
import sys
import os

def resize_logo(input_path):
    # Sizes needed for PWA icons
    sizes = [72, 96, 128, 144, 152, 192, 384, 512]
    
    try:
        # Open the original image
        img = Image.open(input_path)
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
            
        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Create resized versions
        for size in sizes:
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            output_path = f'icon-{size}x{size}.png'
            resized.save(output_path, 'PNG')
            print(f'Created {output_path}')
            
    except Exception as e:
        print(f'Error: {e}')
        return False
    
    return True

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python resize_logo.py <path_to_logo.png>')
    else:
        resize_logo(sys.argv[1])
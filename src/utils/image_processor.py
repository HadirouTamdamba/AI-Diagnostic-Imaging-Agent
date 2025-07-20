"""
Image processing utilities with optimization and validation
"""
import os
import io
import logging
from PIL import Image as PILImage
from PIL.ExifTags import TAGS
from typing import Tuple, Optional, Dict, Any
import numpy as np
from agno.media import Image as AgnoImage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Enhanced image processing with medical image optimization"""
    
    def __init__(self, max_size: int = 5*1024*1024, target_size: Tuple[int, int] = (1024, 1024)):
        self.max_size = max_size
        self.target_size = target_size
    
    def validate_image(self, uploaded_file) -> Dict[str, Any]:
        """Comprehensive image validation"""
        try:
            # Size validation
            if uploaded_file.size > self.max_size:
                return {
                    "valid": False,
                    "error": f"Image size ({uploaded_file.size / (1024*1024):.1f}MB) exceeds limit ({self.max_size / (1024*1024):.1f}MB)"
                }
            
            # Format validation
            image = PILImage.open(uploaded_file)
            if image.format.lower() not in ['jpeg', 'jpg', 'png']:
                return {
                    "valid": False,
                    "error": f"Unsupported format: {image.format}"
                }
            
            # Medical image quality checks
            width, height = image.size
            if width < 100 or height < 100:
                return {
                    "valid": False,
                    "error": "Image resolution too low for medical analysis"
                }
            
            return {
                "valid": True,
                "format": image.format,
                "size": (width, height),
                "mode": image.mode,
                "file_size": uploaded_file.size
            }
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return {
                "valid": False,
                "error": f"Invalid image file: {str(e)}"
            }
    
    def enhance_medical_image(self, image: PILImage.Image) -> PILImage.Image:
        """Apply medical image enhancement techniques"""
        try:
            # Convert to grayscale for better medical analysis if needed
            if image.mode != 'L' and self._is_grayscale_beneficial(image):
                image = image.convert('L')
            
            # Contrast enhancement for medical images
            import PIL.ImageEnhance
            enhancer = PIL.ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Sharpness enhancement
            enhancer = PIL.ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            return image
        except Exception as e:
            logger.warning(f"Enhancement failed, using original: {e}")
            return image
    
    def _is_grayscale_beneficial(self, image: PILImage.Image) -> bool:
        """Determine if grayscale conversion would benefit analysis"""
        # Simple heuristic: if image is mostly grayscale already
        np_image = np.array(image)
        if len(np_image.shape) == 3:
            r, g, b = np_image[:,:,0], np_image[:,:,1], np_image[:,:,2]
            return np.allclose(r, g, atol=30) and np.allclose(g, b, atol=30)
        return False
    
    def optimize_for_analysis(self, uploaded_file) -> Tuple[PILImage.Image, str]:
        """Optimize image for medical analysis"""
        try:
            # Validation
            validation_result = self.validate_image(uploaded_file)
            if not validation_result["valid"]:
                raise ValueError(validation_result["error"])
             
        
            # Load and process image
            image = PILImage.open(uploaded_file)
            
            # Apply medical enhancements
            image = self.enhance_medical_image(image)
            
            # Resize if needed while maintaining aspect ratio
            if image.size[0] > self.target_size[0] or image.size[1] > self.target_size[1]:
                image.thumbnail(self.target_size, PILImage.Resampling.LANCZOS)
            
            # Save optimized image
            temp_path = "temp_optimized_image.png"
            image.save(temp_path, "PNG", optimize=True)
            
            logger.info(f"Image optimized: {validation_result['size']} -> {image.size}")
            
            return image, temp_path
            
        except Exception as e:
            logger.error(f"Image optimization error: {e}")
            raise
    
    def create_agno_image(self, image_path: str) -> AgnoImage:
        """Create AgnoImage object with error handling"""
        try:
            return AgnoImage(filepath=image_path)
        except Exception as e:
            logger.error(f"Failed to create AgnoImage: {e}")
            raise
    
    def cleanup_temp_files(self, *file_paths: str) -> None:
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")
                
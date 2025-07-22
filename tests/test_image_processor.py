"""
Unit tests for image processor functionality
"""
import pytest
import io
from PIL import Image as PILImage
import numpy as np
from src.utils.image_processor import ImageProcessor

class TestImageProcessor: 
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = ImageProcessor()
    
    def create_test_image(self, size=(100, 100), format='PNG'):
        """Create a test image for testing"""
        image = PILImage.new('RGB', size, color='red')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=format)
        img_bytes.seek(0)
        return img_bytes
    
    def test_image_validation_success(self):
        """Test successful image validation"""
        test_file = self.create_test_image()
        test_file.size = 1024  # Mock file size
        
        result = self.processor.validate_image(test_file)
        
        assert result["valid"] is True
        assert "format" in result
        assert "size" in result
    
    def test_image_validation_size_limit(self):
        """Test image size validation"""
        test_file = self.create_test_image()
        test_file.size = 10 * 1024 * 1024  # 10MB
        
        result = self.processor.validate_image(test_file)
        
        assert result["valid"] is False
        assert "size" in result["error"]
    
    def test_image_enhancement(self):
        """Test medical image enhancement"""
        test_image = PILImage.new('RGB', (200, 200), color='gray')
        enhanced = self.processor.enhance_medical_image(test_image)
        
        assert enhanced.size == test_image.size
        assert isinstance(enhanced, PILImage.Image)
    
    def test_optimization_workflow(self):
        """Test complete optimization workflow"""
        test_file = self.create_test_image(size=(2000, 2000))
        test_file.size = 1024
        test_file.name = "test_image.png"
        
        optimized_image, temp_path = self.processor.optimize_for_analysis(test_file)
        
        assert optimized_image.size[0] <= 1024
        assert optimized_image.size[1] <= 1024
        assert temp_path.endswith('.png')
        
        # Cleanup
        self.processor.cleanup_temp_files(temp_path)
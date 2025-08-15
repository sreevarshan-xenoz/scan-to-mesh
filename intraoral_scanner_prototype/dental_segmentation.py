"""
Dental Segmentation - Basic computer vision for dental feature detection
Simplified version of the AI-powered segmentation used in professional systems
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
import open3d as o3d

class DentalSegmentation:
    def __init__(self):
        self.tooth_regions = []
        self.gum_regions = []
        
    def segment_oral_cavity(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Basic segmentation of oral cavity into teeth and gums
        Uses color-based segmentation as a simplified approach
        """
        # Convert to different color spaces for better segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Segment teeth (typically whiter regions)
        teeth_mask = self._segment_teeth(image, hsv, lab)
        
        # Segment gums (typically pinker/redder regions)
        gums_mask = self._segment_gums(image, hsv, lab)
        
        # Clean up masks
        teeth_mask = self._clean_mask(teeth_mask)
        gums_mask = self._clean_mask(gums_mask)
        
        return {
            'teeth': teeth_mask,
            'gums': gums_mask,
            'background': ~(teeth_mask | gums_mask)
        }
    
    def _segment_teeth(self, image: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
        """Segment teeth using color and brightness thresholds"""
        # Teeth are typically bright and have low saturation
        brightness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold for bright regions
        _, bright_mask = cv2.threshold(brightness, 120, 255, cv2.THRESH_BINARY)
        
        # Low saturation in HSV
        _, low_sat_mask = cv2.threshold(hsv[:,:,1], 80, 255, cv2.THRESH_BINARY_INV)
        
        # High lightness in LAB
        _, high_light_mask = cv2.threshold(lab[:,:,0], 140, 255, cv2.THRESH_BINARY)
        
        # Combine masks
        teeth_mask = bright_mask & low_sat_mask & high_light_mask
        
        return teeth_mask.astype(bool)
    
    def _segment_gums(self, image: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
        """Segment gums using color characteristics"""
        # Gums typically have pink/red hue
        hue = hsv[:,:,0]
        sat = hsv[:,:,1]
        
        # Pink/red hue range (accounting for HSV wraparound)
        pink_mask1 = (hue >= 0) & (hue <= 20)  # Red-pink
        pink_mask2 = (hue >= 160) & (hue <= 180)  # Pink-red
        hue_mask = pink_mask1 | pink_mask2
        
        # Moderate saturation
        sat_mask = (sat >= 30) & (sat <= 150)
        
        # Moderate brightness
        brightness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_mask = (brightness >= 60) & (brightness <= 180)
        
        # Combine masks
        gums_mask = hue_mask & sat_mask & bright_mask
        
        return gums_mask.astype(bool)
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean up segmentation mask using morphological operations"""
        # Convert to uint8 for OpenCV operations
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        return mask_uint8 > 0
    
    def detect_individual_teeth(self, teeth_mask: np.ndarray, image: np.ndarray) -> List[Dict]:
        """
        Detect individual teeth within the teeth mask
        Uses connected components and shape analysis
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            teeth_mask.astype(np.uint8) * 255, connectivity=8)
        
        teeth = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            # Get component properties
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by size (teeth should be reasonably sized)
            if area < 100 or area > 5000:
                continue
            
            # Create mask for this tooth
            tooth_mask = (labels == i)
            
            # Get bounding box
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
            
            # Calculate shape features
            contours, _ = cv2.findContours(
                tooth_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                contour = contours[0]
                
                # Shape analysis
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                
                teeth.append({
                    'id': i,
                    'mask': tooth_mask,
                    'centroid': centroids[i],
                    'bounding_box': (x, y, w, h),
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'contour': contour
                })
        
        return teeth
    
    def segment_mesh_regions(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, np.ndarray]:
        """
        Segment 3D mesh into dental regions using geometric features
        Simplified version of 3D segmentation
        """
        if mesh is None or len(mesh.vertices) == 0:
            return {}
        
        vertices = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        
        # Use height (Y coordinate) for basic segmentation
        y_coords = vertices[:, 1]
        
        # Segment based on height (simplified approach)
        # Upper teeth, lower teeth, gums
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        y_range = y_max - y_min
        
        # Define regions based on height
        upper_teeth_mask = y_coords > (y_min + 0.7 * y_range)
        lower_teeth_mask = y_coords < (y_min + 0.3 * y_range)
        gums_mask = ~(upper_teeth_mask | lower_teeth_mask)
        
        # If colors are available, refine segmentation
        if colors is not None:
            # Use color information to improve segmentation
            brightness = np.mean(colors, axis=1)
            
            # Teeth are typically brighter
            bright_mask = brightness > 0.6
            
            # Refine teeth masks
            upper_teeth_mask = upper_teeth_mask & bright_mask
            lower_teeth_mask = lower_teeth_mask & bright_mask
            
            # Gums are the rest
            gums_mask = ~(upper_teeth_mask | lower_teeth_mask)
        
        return {
            'upper_teeth': upper_teeth_mask,
            'lower_teeth': lower_teeth_mask,
            'gums': gums_mask
        }
    
    def analyze_dental_features(self, image: np.ndarray, teeth: List[Dict]) -> Dict:
        """
        Analyze dental features for clinical assessment
        Simplified version of professional analysis
        """
        analysis = {
            'tooth_count': len(teeth),
            'total_tooth_area': sum(tooth['area'] for tooth in teeth),
            'average_tooth_size': np.mean([tooth['area'] for tooth in teeth]) if teeth else 0,
            'teeth_distribution': {},
            'potential_issues': []
        }
        
        if not teeth:
            return analysis
        
        # Analyze tooth distribution
        centroids = np.array([tooth['centroid'] for tooth in teeth])
        
        # Divide into quadrants for distribution analysis
        img_center_x = image.shape[1] / 2
        img_center_y = image.shape[0] / 2
        
        quadrants = {
            'upper_left': np.sum((centroids[:, 0] < img_center_x) & (centroids[:, 1] < img_center_y)),
            'upper_right': np.sum((centroids[:, 0] >= img_center_x) & (centroids[:, 1] < img_center_y)),
            'lower_left': np.sum((centroids[:, 0] < img_center_x) & (centroids[:, 1] >= img_center_y)),
            'lower_right': np.sum((centroids[:, 0] >= img_center_x) & (centroids[:, 1] >= img_center_y))
        }
        
        analysis['teeth_distribution'] = quadrants
        
        # Check for potential issues
        tooth_areas = [tooth['area'] for tooth in teeth]
        if tooth_areas:
            area_std = np.std(tooth_areas)
            area_mean = np.mean(tooth_areas)
            
            # Flag unusually small or large teeth
            for tooth in teeth:
                if tooth['area'] < area_mean - 2 * area_std:
                    analysis['potential_issues'].append(f"Small tooth detected at {tooth['centroid']}")
                elif tooth['area'] > area_mean + 2 * area_std:
                    analysis['potential_issues'].append(f"Large tooth detected at {tooth['centroid']}")
        
        return analysis
    
    def create_segmentation_overlay(self, image: np.ndarray, segments: Dict[str, np.ndarray]) -> np.ndarray:
        """Create colored overlay showing segmentation results"""
        overlay = image.copy()
        
        # Define colors for different segments
        colors = {
            'teeth': (255, 255, 255),  # White
            'gums': (255, 100, 100),   # Light red
            'background': (0, 0, 0)    # Black
        }
        
        # Apply colored overlays
        for segment_name, mask in segments.items():
            if segment_name in colors:
                color = colors[segment_name]
                overlay[mask] = cv2.addWeighted(
                    overlay[mask], 0.7, 
                    np.full_like(overlay[mask], color), 0.3, 0
                )
        
        return overlay
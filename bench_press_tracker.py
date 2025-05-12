import cv2
import numpy as np
import pygame
import time
from scipy.io import wavfile
import os
import json

class BenchPressTracker:
    def __init__(self, video_source=0, mode='live'):
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:  # Fallback if FPS is not available
            self.fps = 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize pygame for sound
        pygame.mixer.init()
        self.beep_sound = pygame.mixer.Sound('beep.wav')
        
        # Initialize weight adjustment sounds - use class's generate_beep method
        self.weight_up_sound = self.generate_beep(880, 0.1)    # Higher pitch for weight increase
        self.weight_down_sound = self.generate_beep(440, 0.1)  # Lower pitch for weight decrease
        
        # Initialize tracking variables
        self.is_tracking = False
        self.rep_count = 0
        self.mode = mode
        self.countdown_active = False
        self.countdown_start = 0
        self.countdown_duration = 3  # seconds
        self.weight = 0  # Initialize weight to 0

        # Fysieke constanten en kalibratie EERST initialiseren
        self.GRAVITY = 9.81  # m/s^2
        self.ACTUAL_BAR_LENGTH_METERS = 2.2 # Standaard Olympische halter
        self.pixels_per_meter = None # Wordt berekend of geladen
        self.bar_length_px = None # Wordt geladen of gemeten (ook al staat het later bij Bar measurement mode, hier alvast None)

        # Reset feedback
        self.reset_message = None
        self.reset_time = None
        self.reset_message_duration = 3.0  # Toon reset melding 3 seconden
        
        # Window visibility toggles
        self.show_bar_detection = False
        self.show_hsv_controls = False
        
        # Initialize debug windows
        self.debug_windows = ["Main", "Bar Detection"]
        for name in self.debug_windows:
            if name == "Bar Detection" and not self.show_bar_detection:
                continue  # Skip creating Bar Detection window if disabled
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(name, 600, 100)  # Position Main window to the right
            cv2.resizeWindow(name, 400, 300)
        
        # Weight tracking
        self.bar_weight = 20
        self.plate_weight = 140
        self.total_weight = self.bar_weight + self.plate_weight
        self.weight_change_time = None
        self.weight_change_direction = None
        self.feedback_duration = 1.0
        
        # Video properties
        self.frame_width = None
        self.frame_height = None
        
        # Playback speed control (1.0 = normal speed, higher = slower)
        self.speed_factor = 1.0
        self.max_speed_factor = 5.0  # Maximum slowdown factor
        
        # Sound feedback
        self.short_pause_sound = self.generate_beep(220, 0.1)  # A3 note for short pause warning
        self.rep_complete_sound = self.generate_beep(880, 0.1)  # A5 note for rep completion
        
        # Bar detection HSV thresholds (for silver/metallic bar)
        self.bar_hsv_lower = np.array([0, 0, 197])  # Updated V min to 197
        self.bar_hsv_upper = np.array([180, 17, 255])  # Updated S max to 17
        
        # ROI selection variables
        self.roi_selecting = False
        self.roi_start_point = None
        self.roi_end_point = None
        self.roi_defined = False
        self.roi_file = "roi_settings.json"  # File to save ROI settings
        
        # NU PAS ROI en kalibratie laden, NA initialisatie van ACTUAL_BAR_LENGTH_METERS
        self.load_roi() 
        
        # Bar tracking stabilization
        self.bar_width_history = []
        self.max_history_length = 10
        self.stabilized_bar_width = 0
        self.max_bar_width = 0  # Keep track of maximum detected width
        self.stabilized_bar_angle = 0
        self.previous_stabilized_angle = 0  # Initialize previous angle
        self.bar_angle_history = []
        self.valid_detection = False  # Flag to track if we have a valid detection
        
        # Bar movement tracking
        self.prev_bar_y = None
        self.movement_state = "UNKNOWN"  # UPWARD, DOWNWARD, or STABLE
        self.y_positions = []
        self.movement_threshold = 0.3  # Verlaagd voor gevoelige detectie
        self.stable_frames = 0
        self.min_stable_frames = 1  # Verlaagd van 2 naar 1 om nog sneller 'stable' te worden
        
        # Added filter for movement detection - gebruik positie als snelheid is onduidelijk
        self.last_position_diff = 0
        self.upward_trend_frames = 0 # Nieuw voor stabielere BOTTOM -> ASCENDING
        self.consecutive_upward_frames_in_bottom = 0 # Nieuw voor BOTTOM -> ASCENDING
        self.downward_frames_in_ascent = 0 # Nieuw voor ASCENDING -> READY stabiliteit
        self.pending_reset_after_failed_ascent = False # Nieuw voor uitgestelde reset
        
        # Rep validatie parameters
        self.min_rep_distance = 40  # Verlaagd van 50 naar 40 voor soepelere validatie
        self.min_phase_duration = 0.25  # Verhoogd van 0.15 naar 0.25 voor stabielere ASCENDING -> COMPLETED
        self.rep_validation_threshold = 0.5  # Fractie van de ROI hoogte die minimaal moet worden afgelegd
        
        # Stable detection buffer
        self.stable_detection_time = None  # Tijd waarop we voor het eerst STABLE detecteerden
        self.min_time_for_stable = 0.1  # Minimale tijd die nodig is om van STABLE naar BOTTOM te gaan
        
        # Debug fields
        self.debug_info = {
            "min_y": float('inf'),
            "max_y": float('-inf'),
            "stable_at": None,
            "transition_blocked_reason": None,
            "last_state_debug": None
        }
        
        # Rep state machine variabelen
        self.rep_state = "READY"  # READY, DESCENDING, BOTTOM, ASCENDING, COMPLETED
        self.state_change_time = time.time()
        self.consecutive_state_frames = 0
        self.state_frames_threshold = 3  # Aantal frames voordat een state overgang wordt geaccepteerd
        self.rep_state_history = []  # Geschiedenis van state overgangen (voor debugging)
        
        # Rep statistics
        self.current_rep_stats = {
            "start_time": None,
            "bottom_time": None,
            "end_time": None,
            "pause_duration": 0,
            "descent_distance": 0, # in pixels
            "ascent_distance": 0,  # in pixels
            "max_descent_speed": 0, # in pixels/s
            "max_ascent_speed": 0,  # in pixels/s
            "avg_descent_speed": 0, # in pixels/s
            "avg_ascent_speed": 0,  # in pixels/s
            "start_y": None,
            "bottom_y": None,
            "end_y": None,
            "_rep_counted_this_cycle": False, 
            # Nieuwe statistieken
            "peak_power_ascent": None, # in Watts
            "avg_power_ascent": None,  # in Watts
            "descent_angles": [],
            "ascent_angles": [],
            "avg_angle_descent": None,
            "std_angle_descent": None,
            "max_abs_angle_descent": None,
            "avg_angle_ascent": None,
            "std_angle_ascent": None,
            "max_abs_angle_ascent": None,
            # Real-time power display velden
            "rt_instant_power": 0,
            "rt_peak_power_this_ascent": 0 
        }
        self.rep_history = []  # List of completed rep statistics
        
        # Speed calculation
        self.speed_buffer = []  # Store recent speed values
        self.speed_buffer_size = 5  # Number of frames for average speed
        self.last_y_for_speed = None
        self.last_time_for_speed = None
        
        # Tracking variables for position and timing
        self.start_y_position = None  # Starting Y position of the bar
        self.bottom_y_position = None  # Lowest Y position during the rep
        
        # Video navigation feedback
        self.nav_feedback = None
        self.nav_feedback_time = 0
        self.nav_feedback_duration = 0.5  # seconds
        
        # Bar measurement mode
        self.measuring_mode = False
        self.measure_start_point = None
        self.measure_end_point = None
        
        # Pause state
        self.paused = False
        self.current_frame = None  # Store the current frame when paused
        
        # Add this line to ensure setup_capture is called with the correct mode
        self.setup_capture()

        # Create HSV trackbars only if HSV Controls window is enabled
        if self.show_hsv_controls:
            self.create_hsv_trackbars()

    def setup_capture(self):
        """Setup video capture based on mode"""
        if self.mode == 'debug':
            video_path = 'movies/test_bench.mp4'
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Get FPS from video file
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if self.fps <= 0:  # Fallback if FPS is not available
                self.fps = 60  # Changed from 30 to 60 as default
                print("Warning: Could not detect FPS from video file, using default 60 FPS")
            else:
                print(f"Detected video FPS: {self.fps}")
                if self.fps != 60:
                    print(f"Warning: Video FPS ({self.fps}) is different from expected 60 FPS")
            
            # Set window size based on video dimensions
            cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Main", self.frame_width, self.frame_height)
            
            # Set up mouse callback for ROI selection
            cv2.setMouseCallback("Main", self.handle_mouse_events)
            print("Mouse callback set for ROI selection. Click and drag to define region.")
            
            self.is_tracking = True  # Auto-start tracking in debug mode
            print(f"Debug Mode: Video loaded - {self.frame_width}x{self.frame_height} at {self.fps}fps")
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.fps = 30  # Set default FPS for live camera
            
            # Set up mouse callback for ROI selection
            cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Main", self.handle_mouse_events)
            print("Mouse callback set for ROI selection. Click and drag to define region.")
            
            print("Live Mode: Camera initialized")

    def switch_mode(self):
        """Switch between live and debug modes"""
        self.cleanup_capture()
        self.mode = 'debug' if self.mode == 'live' else 'live'
        self.setup_capture()
        print(f"Switched to {self.mode.upper()} mode")

    def cleanup_capture(self):
        """Cleanup current capture"""
        if self.cap is not None:
            self.cap.release()
            
    def set_weight(self, bar_weight, plate_weight):
        """Set the current weight being lifted"""
        self.bar_weight = bar_weight
        self.plate_weight = plate_weight
        self.total_weight = bar_weight + plate_weight

    def draw_metrics(self, frame):
        """Draw performance metrics and tracking visualization"""
        # Draw speed control slider in debug mode
        if self.mode == 'debug':
            slider_x = 10
            slider_y = frame.shape[0] - 50
            slider_width = 200
            slider_height = 30
            
            # Draw slider background
            cv2.rectangle(frame, 
                         (slider_x, slider_y),
                         (slider_x + slider_width, slider_y + slider_height),
                         (0, 0, 0),
                         -1)
            
            # Draw slider track
            cv2.line(frame,
                     (slider_x + 10, slider_y + slider_height//2),
                     (slider_x + slider_width - 10, slider_y + slider_height//2),
                     (128, 128, 128),
                     2)
            
            # Calculate slider position
            slider_pos = int(((self.speed_factor - 1.0) / (self.max_speed_factor - 1.0)) * (slider_width - 20)) + slider_x + 10
            
            # Draw slider handle
            cv2.circle(frame,
                      (slider_pos, slider_y + slider_height//2),
                      8,
                      (0, 255, 0),
                      -1)
            
            # Draw speed text
            speed_text = f"Speed: {1.0/self.speed_factor:.1f}x"
            cv2.putText(frame,
                       speed_text,
                       (slider_x + slider_width + 10, slider_y + slider_height - 8),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6,
                       (255, 255, 255),
                       1)
            
    def generate_beep(self, frequency, duration):
        """Generate a simple beep sound"""
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        samples = np.sin(2 * np.pi * frequency * t)
        samples = np.int16(samples * 32767)
        # Convert to stereo by duplicating the mono signal
        stereo_samples = np.column_stack((samples, samples))
        return pygame.sndarray.make_sound(stereo_samples)

    def adjust_weight(self, amount):
        """Adjust the plate weight by the given amount"""
        old_weight = self.total_weight
        self.plate_weight = max(0, self.plate_weight + amount)  # Prevent negative weight
        self.total_weight = self.bar_weight + self.plate_weight
        
        if self.total_weight != old_weight:  # Only trigger feedback if weight actually changed
            self.weight_change_time = time.time()
            self.weight_change_direction = 'increase' if amount > 0 else 'decrease'
            
            # Play appropriate sound
            if amount > 0:
                self.weight_up_sound.play()
            else:
                self.weight_down_sound.play()
            
            print(f"Total weight adjusted to: {self.total_weight}kg")

    def draw_weight_display(self, frame):
        """Draw the current weight in the top right corner with visual feedback"""
        weight_text = f"{self.total_weight}kg"
        
        # Get text size to position it properly
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(weight_text, font, font_scale, thickness)[0]
        
        # Position in top right with some padding
        x = frame.shape[1] - text_size[0] - 20
        y = text_size[1] + 20
        
        # Determine text color based on weight change
        if self.weight_change_time is not None:
            elapsed_time = time.time() - self.weight_change_time
            if elapsed_time < self.feedback_duration:
                # Calculate fade effect (1.0 to 0.0 over feedback_duration)
                fade = max(0, 1 - (elapsed_time / self.feedback_duration))
                
                if self.weight_change_direction == 'increase':
                    # Fade from green to red
                    green = int(255 * fade)
                    text_color = (0, green, 255 - green)  # BGR format
                    
                    # Add up arrow with fade effect
                    arrow_y = y - text_size[1] - 30
                    arrow_points = np.array([[x + text_size[0]//2 - 20, arrow_y + 20],
                                           [x + text_size[0]//2, arrow_y],
                                           [x + text_size[0]//2 + 20, arrow_y + 20]])
                    cv2.fillPoly(frame, [arrow_points], (0, green, 0))
                    
                else:  # decrease
                    # Fade from bright red to normal red
                    red = int(255 * (1 - fade/2))  # Less intense fade for decrease
                    text_color = (0, 0, red)  # BGR format
                    
                    # Add down arrow with fade effect
                    arrow_y = y + 10
                    arrow_points = np.array([[x + text_size[0]//2 - 20, arrow_y],
                                           [x + text_size[0]//2, arrow_y + 20],
                                           [x + text_size[0]//2 + 20, arrow_y]])
                    cv2.fillPoly(frame, [arrow_points], (0, 0, red))
            else:
                self.weight_change_time = None
                text_color = (0, 0, 255)  # Default red
        else:
            text_color = (0, 0, 255)  # Default red
        
        # Draw the weight text
        cv2.putText(frame, weight_text, (x, y),
                    font, font_scale, text_color, thickness)
        
        # Draw small reminder text for controls with increased size and spacing
        reminder_scale = 0.7  # Increased from 0.5
        reminder_y = y + 40  # Increased from 30
        cv2.putText(frame, "+/- : 10kg", (x, reminder_y),
                    font, reminder_scale, (128, 128, 128), 1)
        cv2.putText(frame, "o/p : 2.5kg", (x, reminder_y + 30),  # Increased from 20
                    font, reminder_scale, (128, 128, 128), 1)

    def draw_mode_indicator(self, frame):
        """Draw current mode indicator"""
        mode_text = f"{self.mode.upper()} MODE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(mode_text, font, font_scale, thickness)[0]
        
        # Position in top left corner
        x = 10
        y = text_size[1] + 10
        
        # Draw mode text
        color = (0, 255, 0) if self.mode == 'live' else (0, 165, 255)  # Green for live, orange for debug
        cv2.putText(frame, mode_text, (x, y),
                   font, font_scale, color, thickness)
        
        # Draw rep counter below mode indicator
        rep_text = f"Reps: {self.rep_count}"
        rep_scale = 1.0
        rep_thickness = 2
        
        # Position rep counter below mode indicator with reduced spacing
        rep_x = x
        rep_y = y + text_size[1] + 20  # Reduced from 70 to 20
        
        # Draw rep counter text
        cv2.putText(frame, rep_text, (rep_x, rep_y),
                   font, rep_scale, (0, 255, 0), rep_thickness)
        
        # Draw angle below rep counter
        if hasattr(self, 'stabilized_bar_angle'):
            angle_text = f"Angle: {self.stabilized_bar_angle:.1f}"
            angle_y = rep_y + 30  # Position below rep counter
            cv2.putText(frame, angle_text, (rep_x, angle_y),
                       font, rep_scale, (0, 255, 255), rep_thickness)
        
        # Draw timer at the top of the frame
        if self.mode == 'debug':
            # Get current frame number
            frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Calculate elapsed time in seconds
            elapsed_seconds = frame_number / self.fps
            # Format time as MM:SS
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
            time_text = f"{minutes:02d}:{seconds:02d}"
            
            # Get text size for timer
            time_scale = 1.5  # Increased scale for better visibility
            time_thickness = 3
            time_text_size = cv2.getTextSize(time_text, font, time_scale, time_thickness)[0]
            
            # Position timer at the top of the frame
            time_x = (frame.shape[1] - time_text_size[0]) // 2  # Center horizontally
            time_y = 20 + time_text_size[1]  # 20px from top
            
            # Draw background for timer
            padding = 10
            cv2.rectangle(frame,
                         (time_x - padding, time_y - time_text_size[1] - padding),
                         (time_x + time_text_size[0] + padding, time_y + padding),
                         (0, 0, 0),
                         -1)
            
            # Draw timer text
            cv2.putText(frame, time_text, (time_x, time_y),
                       font, time_scale, (255, 255, 255), time_thickness)

    def create_hsv_trackbars(self):
        """Create trackbars for adjusting HSV thresholds"""
        # Only create HSV Controls window if enabled
        if not self.show_hsv_controls:
            return
            
        cv2.namedWindow("HSV Controls")
        
        # Create trackbars grouped by component (H, S, V)
        cv2.createTrackbar("H min", "HSV Controls", self.bar_hsv_lower[0], 180, lambda x: self.update_hsv(x, 0, 'lower'))
        cv2.createTrackbar("H max", "HSV Controls", self.bar_hsv_upper[0], 180, lambda x: self.update_hsv(x, 0, 'upper'))
        
        cv2.createTrackbar("S min", "HSV Controls", self.bar_hsv_lower[1], 255, lambda x: self.update_hsv(x, 1, 'lower'))
        cv2.createTrackbar("S max", "HSV Controls", self.bar_hsv_upper[1], 255, lambda x: self.update_hsv(x, 1, 'upper'))
        
        cv2.createTrackbar("V min", "HSV Controls", self.bar_hsv_lower[2], 255, lambda x: self.update_hsv(x, 2, 'lower'))
        cv2.createTrackbar("V max", "HSV Controls", self.bar_hsv_upper[2], 255, lambda x: self.update_hsv(x, 2, 'upper'))

    def update_hsv(self, value, index, bound):
        """Update HSV threshold values based on trackbar position"""
        if bound == 'lower':
            self.bar_hsv_lower[index] = value
        else:
            self.bar_hsv_upper[index] = value
            
    def adjust_hsv_value(self, index, bound, amount):
        """Adjust HSV value by the given amount and update trackbar"""
        if bound == 'lower':
            current = self.bar_hsv_lower[index]
            max_val = 180 if index == 0 else 255  # H max is 180, S and V max are 255
            new_val = max(0, min(max_val, current + amount))
            self.bar_hsv_lower[index] = new_val
            # Update trackbar
            trackbar_name = ["H min", "S min", "V min"][index]
            cv2.setTrackbarPos(trackbar_name, "HSV Controls", new_val)
            print(f"{trackbar_name} adjusted to {new_val}")
        else:
            current = self.bar_hsv_upper[index]
            max_val = 180 if index == 0 else 255  # H max is 180, S and V max are 255
            new_val = max(0, min(max_val, current + amount))
            self.bar_hsv_upper[index] = new_val
            # Update trackbar
            trackbar_name = ["H max", "S max", "V max"][index]
            cv2.setTrackbarPos(trackbar_name, "HSV Controls", new_val)
            print(f"{trackbar_name} adjusted to {new_val}")

    def calculate_bar_metrics(self, frame):
        """Calculate bar position using HSV color detection"""
        # If ROI is defined, only process that region
        if self.roi_defined:
            try:
                # Extract region of interest
                x1, y1 = self.roi_start_point
                x2, y2 = self.roi_end_point
                roi = frame[y1:y2, x1:x2]
                
                # Convert ROI to HSV
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Create mask for bar detection
                mask = cv2.inRange(hsv, self.bar_hsv_lower, self.bar_hsv_upper)
                
                # Store original mask for debugging
                original_mask = mask.copy()
                
                # Apply morphological operations with hardcoded values
                # Skip closing operation (Close Width = 0)
                
                # Then remove small noise
                kernel_open = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                
                # Apply horizontal dilation
                kernel_dilate = np.ones((1, 19), np.uint8)
                mask = cv2.dilate(mask, kernel_dilate, iterations=4)
                
                # Create a debug composite for ROI only
                roi_height, roi_width = roi.shape[:2]
                debug_composite = np.zeros((roi_height, roi_width * 2), dtype=np.uint8)
                debug_composite[:, :roi_width] = original_mask
                debug_composite[:, roi_width:] = mask
                
                # Add labels
                cv2.putText(debug_composite, "Original", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(debug_composite, "Processed", (roi_width + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show debug composite only if bar detection window is enabled
                if self.show_bar_detection:
                    cv2.imshow("Bar Detection", debug_composite)
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Reset valid detection flag
                self.valid_detection = False
                
                # Add debug info
                contour_areas = [cv2.contourArea(c) for c in contours]
                #print(f"Found {len(contours)} contours, areas: {contour_areas}")
                
                # Filter contours for bar-like objects
                bar_candidates = []
                min_area = 200
                for contour in contours:
                    # Skip very small contours
                    if cv2.contourArea(contour) < min_area:
                        continue
                        
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate aspect ratio (width/height)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    # More permissive criteria for bar segments, but require a minimum aspect ratio
                    if aspect_ratio > 2.0 or (w > 100 and aspect_ratio > 1.5):
                        bar_candidates.append((contour, w, aspect_ratio, y, x))
                
                # Add debug info
                #print(f"Bar candidates: {len(bar_candidates)}")
                
                # Process bar candidates if we found any
                if bar_candidates:
                    # Variables to store current detection
                    x, y, w, h = 0, 0, 0, 0
                    bar_angle = 0
                    
                    # If we have multiple candidates, try to find ones that are horizontally aligned
                    if len(bar_candidates) > 1:
                        # Group candidates that are within a certain vertical range
                        groups = []
                        bar_candidates.sort(key=lambda x: x[3])  # Sort by y-position
                        current_group = [bar_candidates[0]]
                        
                        # More permissive vertical grouping (30px instead of 20px)
                        vertical_tolerance = 30
                        
                        for i in range(1, len(bar_candidates)):
                            # If this candidate is close vertically to the previous one, add to current group
                            if abs(bar_candidates[i][3] - bar_candidates[i-1][3]) < vertical_tolerance:
                                current_group.append(bar_candidates[i])
                            else:
                                # Start a new group
                                groups.append(current_group)
                                current_group = [bar_candidates[i]]
                        
                        # Add the last group
                        groups.append(current_group)
                        
                        # Find the group with the most combined width (likely the bar)
                        if groups:
                            best_group = max(groups, key=lambda g: sum(c[1] for c in g))
                            
                            # Combine all contours in the best group into a single contour
                            combined_contour = np.vstack([c[0] for c in best_group])
                            
                            # Get bounding box for the combined contour
                            x, y, w, h = cv2.boundingRect(combined_contour)
                            
                            # Calculate orientation, constraining to mostly horizontal
                            bar_angle = self.calculate_constrained_angle(combined_contour)
                            
                            # Mark as valid detection
                            self.valid_detection = True
                        else:
                            # Fallback to the largest contour if grouping fails
                            largest_contour = max(contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(largest_contour)
                            bar_angle = self.calculate_constrained_angle(largest_contour)
                            self.valid_detection = True
                    else:
                        # Single candidate - use its bounding rectangle
                        best_contour = bar_candidates[0][0]
                        x, y, w, h = cv2.boundingRect(best_contour)
                        bar_angle = self.calculate_constrained_angle(best_contour)
                        self.valid_detection = True
                    
                    if self.valid_detection:
                        # Update maximum bar width if we found a larger value
                        if 300 <= w <= 800 and w > self.max_bar_width:
                            self.max_bar_width = w
                        
                        # If we don't have a maximum width yet, initialize with current width
                        if self.max_bar_width == 0:
                            self.max_bar_width = w
                        
                        # Stabilize angle measurement
                        self.bar_angle_history.append(bar_angle)
                        if len(self.bar_angle_history) > self.max_history_length:
                            self.bar_angle_history.pop(0)
                        
                        # Get mean angle handling wraparound
                        sum_sin = sum(np.sin(np.radians(angle)) for angle in self.bar_angle_history)
                        sum_cos = sum(np.cos(np.radians(angle)) for angle in self.bar_angle_history)
                        
                        # Calculate the new angle
                        new_angle = np.degrees(np.arctan2(sum_sin, sum_cos))
                        
                        # Check if we have a previous stabilized angle and the change is too large
                        if hasattr(self, 'previous_stabilized_angle'):
                            angle_change = abs(new_angle - self.previous_stabilized_angle)
                            # If angle change is too large (more than 0.5 degrees), ignore it
                            if angle_change > 0.5:
                                # Use the previous angle instead
                                new_angle = self.previous_stabilized_angle
                        
                        # Update the stabilized angle and store for next frame
                        self.stabilized_bar_angle = new_angle
                        self.previous_stabilized_angle = new_angle
                        
                        # Further constrain angle to avoid vertical orientations
                        if abs(self.stabilized_bar_angle - 90) < abs(self.stabilized_bar_angle):
                            self.stabilized_bar_angle = 0
                            self.previous_stabilized_angle = 0
                        
                        # Calculate center of the bar
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Use the actual detected width rather than max width for better accuracy
                        bar_width = w if 300 <= w <= 800 else self.max_bar_width
                        
                        # Calculate the endpoints directly based on the bar's actual position
                        x1 = x
                        y1 = y + h // 2
                        x2 = x + w
                        y2 = y + h // 2
                        
                        # Adjust endpoints with angle if needed
                        if abs(self.stabilized_bar_angle) > 0.5:
                            half_width = bar_width // 2
                            angle_rad = np.radians(self.stabilized_bar_angle)
                            
                            # Calculate the endpoints with angle
                            x1 = int(center_x - half_width * np.cos(angle_rad))
                            y1 = int(center_y - half_width * np.sin(angle_rad))
                            x2 = int(center_x + half_width * np.cos(angle_rad))
                            y2 = int(center_y + half_width * np.sin(angle_rad))
                        
                        # Draw the stabilized angled line along the bar
                        line_thickness = 3
                        line_color = (0, 255, 0)  # Green line
                        
                        # Translate ROI coordinates to frame coordinates for drawing
                        frame_x1 = x1 + self.roi_start_point[0]
                        frame_y1 = y1 + self.roi_start_point[1]
                        frame_x2 = x2 + self.roi_start_point[0]
                        frame_y2 = y2 + self.roi_start_point[1]
                        frame_center_x = center_x + self.roi_start_point[0]
                        frame_center_y = center_y + self.roi_start_point[1]
                        
                        # Debug info - print coordinates
                        #print(f"Bar position: center=({frame_center_x}, {frame_center_y})")
                        
                        # Track bar movement (using vertical position)
                        if self.prev_bar_y is not None:
                            # Calculate y_diff (negative is up, positive is down)
                            y_diff = frame_center_y - self.prev_bar_y
                            
                            # Direct bijhouden van het verschil voor backup detectie
                            self.last_position_diff = y_diff
                            
                            # Store position history
                            self.y_positions.append(frame_center_y)
                            if len(self.y_positions) > self.max_history_length:
                                self.y_positions.pop(0)
                            
                            # Calculate average movement over recent frames for smoothness
                            if len(self.y_positions) >= 3:
                                # Get the average movement over the last few frames
                                recent_diffs = []
                                for i in range(1, min(5, len(self.y_positions))):
                                    recent_diffs.append(self.y_positions[-i] - self.y_positions[-i-1])
                                avg_movement = sum(recent_diffs) / len(recent_diffs)
                                
                                # Determine movement state
                                prev_state = self.movement_state
                                if abs(avg_movement) < self.movement_threshold:
                                    self.stable_frames += 1
                                    if self.stable_frames >= self.min_stable_frames:
                                        self.movement_state = "STABLE"
                                        
                                        # Record time when we first detect stable
                                        if prev_state != "STABLE" and self.stable_detection_time is None:
                                            self.stable_detection_time = time.time()
                                            print(f"Started STABLE detection at {self.stable_detection_time}")
                                else:
                                    self.stable_frames = 0
                                    self.stable_detection_time = None  # Reset stable detection time
                                    
                                    if avg_movement < 0:  # Remember negative is upward
                                        self.movement_state = "UPWARD"
                                    else:
                                        self.movement_state = "DOWNWARD"
                                
                                # Force movement state based on direct position difference when avg_movement is small
                                # Verhogen van 2* naar 3* self.movement_threshold voor y_diff
                                if abs(avg_movement) < 2*self.movement_threshold and abs(y_diff) > 3*self.movement_threshold:
                                    if y_diff < 0:  # Moving upward
                                        self.movement_state = "UPWARD"
                                        print(f"Force UPWARD state - direct y_diff: {y_diff:.2f}, avg: {avg_movement:.2f}")
                                    elif y_diff > 0:  # Moving downward
                                        self.movement_state = "DOWNWARD"
                                        print(f"Force DOWNWARD state - direct y_diff: {y_diff:.2f}, avg: {avg_movement:.2f}")
                                
                                # Debug when movement state changes
                                if prev_state != self.movement_state:
                                    print(f"Movement state changed: {prev_state} -> {self.movement_state}")
                                    print(f"  Avg movement: {avg_movement:.2f}, Direct y_diff: {y_diff:.2f}, Stable frames: {self.stable_frames}")
                                
                                # Debug current values
                                print(f"Movement: {self.movement_state}, Avg: {avg_movement:.2f}, Direct: {y_diff:.2f}, Y-pos: {frame_center_y}")
                                
                                # Update the rep state machine based on movement
                                try:
                                    self.update_rep_state(frame_center_y)
                                except Exception as e:
                                    print(f"Error in update_rep_state: {e}")
                        else:
                            # Initialize movement state to STABLE when we first detect the bar
                            self.movement_state = "STABLE"
                            print("Initial detection - setting movement state to STABLE")
                        
                        # Update previous position
                        self.prev_bar_y = frame_center_y
                        
                        # Draw using frame coordinates
                        cv2.line(frame, (frame_x1, frame_y1), (frame_x2, frame_y2), line_color, line_thickness)
                        
                        # Add small markers at the ends of the bar
                        marker_size = 5
                        cv2.circle(frame, (frame_x1, frame_y1), marker_size, line_color, -1)
                        cv2.circle(frame, (frame_x2, frame_y2), marker_size, line_color, -1)
                        
                        # Determine movement state color
                        movement_color = (255, 255, 255)  # Default white
                        if self.movement_state == "UPWARD":
                            movement_color = (0, 255, 0)  # Green for upward
                        elif self.movement_state == "DOWNWARD":
                            movement_color = (0, 0, 255)  # Red for downward
                        elif self.movement_state == "STABLE":
                            movement_color = (255, 255, 0)  # Cyan for stable
                        
                        # Display rep state
                        try:
                            # Only use attributes if they exist
                            if hasattr(self, 'rep_state'):
                                state_color = (255, 255, 255)  # Default white
                                if self.rep_state == "READY":
                                    state_color = (0, 255, 255)  # Yellow
                                elif self.rep_state == "DESCENDING":
                                    state_color = (0, 0, 255)  # Red
                                elif self.rep_state == "BOTTOM":
                                    state_color = (255, 0, 0)  # Blue
                                elif self.rep_state == "ASCENDING":
                                    state_color = (0, 255, 0)  # Green
                                elif self.rep_state == "COMPLETED":
                                    state_color = (255, 255, 255)  # White
                                
                                rep_state_text = f"Rep State: {self.rep_state}"
                                cv2.putText(frame, rep_state_text, 
                                           (frame_center_x - 100, frame_center_y - 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
                        except Exception as e:
                            print(f"Error in rep state display: {e}")
                        
                        # Display rep count
                        # rep_count_text = f"Reps: {self.rep_count}"
                        # cv2.putText(frame, rep_count_text,
                        #            (frame_center_x - 100, frame_center_y - 60),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    else:
                        # Reset tracking when no detection
                        self.prev_bar_y = None
                        self.movement_state = "UNKNOWN"
                        self.y_positions = []
                        self.stable_frames = 0
                        print("Lost detection - resetting tracking variables")
            except Exception as e:
                print(f"Error in bar detection: {e}")
                # Draw the error on the frame
                cv2.putText(frame, f"Bar detection error: {str(e)}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def calculate_constrained_angle(self, contour):
        """Calculate the orientation angle of a contour, properly constrained for a barbell"""
        try:
            # For barbells, we can also try a simpler approach using the bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # If the contour is wide enough, use minimum area rectangle for more accurate angle
            if cv2.contourArea(contour) > 1000:
                # Get the minimum area rectangle
                rect = cv2.minAreaRect(contour)
                width = rect[1][0]
                height = rect[1][1]
                angle = rect[2]
                
                # minAreaRect returns angles in the range [-90, 0)
                # Adjust the angle based on the rectangle dimensions
                if width < height:
                    angle = angle - 90
                
                # Normalize angle to [-15, 15] range for barbells (they're usually close to horizontal)
                while angle < -15:
                    angle += 180
                while angle > 15:
                    angle -= 180
                    
                return angle
            
            # Fallback to moments method with better constraints
            moments = cv2.moments(contour)
            
            if moments['m00'] != 0:
                # These moments can be used to find the orientation
                mu20 = moments['mu20'] / moments['m00']
                mu11 = moments['mu11'] / moments['m00']
                mu02 = moments['mu02'] / moments['m00']
                
                # Calculate the angle
                if abs(mu20 - mu02) < 1e-5:  # Almost equal
                    return 0  # Force horizontal
                
                # Calculate the angle
                theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
                angle_deg = np.degrees(theta)
                
                # Normalize angle to [-15, 15] range for barbells (they're usually close to horizontal)
                while angle_deg < -15:
                    angle_deg += 180
                while angle_deg > 15:
                    angle_deg -= 180
                
                return angle_deg
            else:
                return 0
        except:
            # If any error occurs, return 0 (horizontal)
            return 0

    def handle_mouse_events(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection and bar measurement"""
        # Debug output to confirm events are being received
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Mouse down at ({x}, {y})")
        elif event == cv2.EVENT_LBUTTONUP:
            print(f"Mouse up at ({x}, {y})")
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            print(f"Mouse dragging to ({x}, {y})")
            
        # Check if we're in measurement mode
        if self.measuring_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                # Start drawing measurement line
                self.measure_start_point = (x, y)
                self.measure_end_point = (x, y)
                print(f"Started bar measurement at {self.measure_start_point}")
            
            elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
                # Update measurement end point during drag
                self.measure_end_point = (x, y)
            
            elif event == cv2.EVENT_LBUTTONUP and self.measure_start_point:
                # Finish drawing measurement line
                self.measure_end_point = (x, y)
                print(f"Finished bar measurement from {self.measure_start_point} to {self.measure_end_point}")
                
                # Calculate length in pixels
                dx = self.measure_end_point[0] - self.measure_start_point[0]
                dy = self.measure_end_point[1] - self.measure_start_point[1]
                self.bar_length_px = np.sqrt(dx**2 + dy**2)
                print(f"Measured bar length: {self.bar_length_px:.1f} pixels")
                
                if self.bar_length_px > 0 and self.ACTUAL_BAR_LENGTH_METERS > 0:
                    self.pixels_per_meter = self.bar_length_px / self.ACTUAL_BAR_LENGTH_METERS
                    print(f"Calibration: {self.pixels_per_meter:.2f} pixels per meter (bar length: {self.ACTUAL_BAR_LENGTH_METERS}m)")
                    # Sla de bijgewerkte kalibratie op, zelfs als de ROI niet is gewijzigd
                    if self.roi_defined: # Sla alleen op als er een ROI is om mee op te slaan
                        self.save_roi() 
                    else:
                        print("Calibration updated, but ROI is not defined. Define an ROI to save calibration with it.")
                else:
                    self.pixels_per_meter = None
                    print("Warning: Could not calculate pixels_per_meter. Bar length in pixels or actual bar length is zero.")
                # Reset measurement points for next time, but keep self.measuring_mode active
                self.measure_start_point = None 
                self.measure_end_point = None
                # We blijven in measuring_mode totdat de gebruiker 'b' nogmaals drukt.
            return # Belangrijk: return hier om niet per ongeluk ROI selectie te triggeren
            
        # Handle ROI selection (only if not in measurement mode)
        if event == cv2.EVENT_LBUTTONDOWN and not self.roi_selecting:
            # Start drawing ROI
            self.roi_selecting = True
            self.roi_defined = False
            self.roi_start_point = (x, y)
            self.roi_end_point = (x, y)
            print(f"Started ROI selection at {self.roi_start_point}")
        
        elif event == cv2.EVENT_MOUSEMOVE and self.roi_selecting:
            # Update ROI end point during drag
            self.roi_end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP and self.roi_selecting:
            # Finish drawing ROI
            self.roi_selecting = False
            self.roi_end_point = (x, y)
            print(f"Finished ROI selection from {self.roi_start_point} to {self.roi_end_point}")
            
            # Ensure coordinates are in proper order (top-left to bottom-right)
            x1, y1 = self.roi_start_point
            x2, y2 = self.roi_end_point
            
            # Swap if needed
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            # Update coordinates
            self.roi_start_point = (x1, y1)
            self.roi_end_point = (x2, y2)
            
            # Ensure minimal size
            if (self.roi_end_point[0] - self.roi_start_point[0] < 50 or 
                self.roi_end_point[1] - self.roi_start_point[1] < 20):
                # If ROI is too small, reset it
                self.roi_defined = False
                print("ROI too small, please draw a larger region")
            else:
                self.roi_defined = True
                print(f"ROI defined: from {self.roi_start_point} to {self.roi_end_point}")
                # Save the ROI for future use
                self.save_roi()

    def save_roi(self):
        """Save ROI coordinates to file"""
        if self.roi_defined and self.roi_start_point and self.roi_end_point:
            roi_data = {
                "start_x": self.roi_start_point[0],
                "start_y": self.roi_start_point[1],
                "end_x": self.roi_end_point[0],
                "end_y": self.roi_end_point[1],
                "bar_length_px": self.bar_length_px, # Voeg gemeten bar lengte toe
                "pixels_per_meter": self.pixels_per_meter, # Voeg kalibratie toe
                "actual_bar_length_meters": self.ACTUAL_BAR_LENGTH_METERS # Ook de gebruikte referentielengte
            }
            try:
                with open(self.roi_file, 'w') as f:
                    json.dump(roi_data, f, indent=4)
                print(f"ROI and calibration saved to {self.roi_file}")
            except Exception as e:
                print(f"Error saving ROI and calibration: {e}")
    
    def load_roi(self):
        """Load ROI coordinates from file"""
        if os.path.exists(self.roi_file):
            try:
                with open(self.roi_file, 'r') as f:
                    roi_data = json.load(f)
                
                # Set ROI coordinates
                self.roi_start_point = (roi_data["start_x"], roi_data["start_y"])
                self.roi_end_point = (roi_data["end_x"], roi_data["end_y"])
                self.roi_defined = True
                print(f"ROI loaded from {self.roi_file}: {self.roi_start_point} to {self.roi_end_point}")

                # Laad kalibratie data indien aanwezig
                self.bar_length_px = roi_data.get("bar_length_px")
                self.pixels_per_meter = roi_data.get("pixels_per_meter")
                loaded_actual_bar_length = roi_data.get("actual_bar_length_meters")

                if self.pixels_per_meter is not None:
                    print(f"Calibration loaded: {self.pixels_per_meter:.2f} px/m (measured bar: {self.bar_length_px:.1f}px for {loaded_actual_bar_length}m bar)")
                    # Optioneel: check of de geladen ACTUAL_BAR_LENGTH_METERS overeenkomt met de huidige instelling
                    if loaded_actual_bar_length is not None and abs(loaded_actual_bar_length - self.ACTUAL_BAR_LENGTH_METERS) > 0.01:
                        print(f"Warning: Loaded calibration used an actual bar length of {loaded_actual_bar_length}m, but current setting is {self.ACTUAL_BAR_LENGTH_METERS}m. Recalibration might be needed if bar type changed.")
                    elif loaded_actual_bar_length is None:
                        print(f"Warning: Old calibration format in {self.roi_file}, actual_bar_length_meters not found. Assuming it was for {self.ACTUAL_BAR_LENGTH_METERS}m.")
                else:
                    print("No calibration data (pixels_per_meter) found in ROI file or it was null.")

            except Exception as e:
                print(f"Error loading ROI or calibration: {e}")
                self.roi_defined = False
                self.pixels_per_meter = None # Zorg dat kalibratie ook gereset wordt bij laadfout
        else:
            print("No saved ROI file found.")
            self.roi_defined = False
            self.pixels_per_meter = None

    def show_navigation_feedback(self, direction):
        """Show a brief visual feedback for navigation"""
        self.nav_feedback = direction  # 'forward' or 'backward'
        self.nav_feedback_time = time.time()
        
    def draw_navigation_feedback(self, frame):
        """Draw navigation feedback arrows if active"""
        if self.nav_feedback is None or time.time() - self.nav_feedback_time > self.nav_feedback_duration:
            self.nav_feedback = None
            return
            
        # Calculate remaining time as a percentage
        elapsed = time.time() - self.nav_feedback_time
        opacity = 1.0 - (elapsed / self.nav_feedback_duration)
        
        # Create an overlay for the arrow
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw a large arrow
        if self.nav_feedback == 'forward':
            # Draw right arrow
            arrow_points = np.array([
                [w//2 + 50, h//2],
                [w//2 - 50, h//2 - 50],
                [w//2 - 50, h//2 + 50]
            ], np.int32)
            color = (0, 255, 0)  # Green for forward
        else:
            # Draw left arrow
            arrow_points = np.array([
                [w//2 - 50, h//2],
                [w//2 + 50, h//2 - 50],
                [w//2 + 50, h//2 + 50]
            ], np.int32)
            color = (0, 165, 255)  # Orange for backward
            
        # Draw the filled arrow
        cv2.fillPoly(overlay, [arrow_points], color)
        
        # Apply the overlay with current opacity
        alpha = 0.7 * opacity
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_hsv_controls_info(self, frame):
        """Draw information about the HSV keyboard controls"""
        # Removed all HSV control text display from the frame
        pass

    def draw_measurement_line(self, frame):
        """Draw the bar measurement line and show its length"""
        if not self.measuring_mode:
            return
            
        # If measurement points exist, draw the line
        if self.measure_start_point and self.measure_end_point:
            # Draw the line with brighter color and thicker
            line_color = (0, 255, 255)  # Bright yellow
            cv2.line(frame, self.measure_start_point, self.measure_end_point, line_color, 3)  # Increased thickness
            
            # Draw more visible markers at the endpoints
            cv2.circle(frame, self.measure_start_point, 8, line_color, -1)  # Increased size
            cv2.circle(frame, self.measure_end_point, 8, line_color, -1)  # Increased size
            
            # Calculate current length
            dx = self.measure_end_point[0] - self.measure_start_point[0]
            dy = self.measure_end_point[1] - self.measure_start_point[1]
            current_length = np.sqrt(dx**2 + dy**2)
            
            # Show length above the line with background
            midpoint_x = (self.measure_start_point[0] + self.measure_end_point[0]) // 2
            midpoint_y = (self.measure_start_point[1] + self.measure_end_point[1]) // 2 - 20
            length_text = f"Length: {current_length:.1f} px"
            
            # Add background for better visibility
            text_size = cv2.getTextSize(length_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, 
                         (midpoint_x - text_size[0]//2 - 5, midpoint_y - text_size[1] - 5),
                         (midpoint_x + text_size[0]//2 + 5, midpoint_y + 5),
                         (0, 0, 0),
                         -1)
            
            cv2.putText(frame, length_text, 
                       (midpoint_x - text_size[0]//2, midpoint_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
        # Show instructions with more visibility
        instructions = "Click and drag to measure bar length - Press 'b' to exit"
        cv2.putText(frame, instructions, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # If a measurement is saved OR a calibration is loaded, display it at the bottom
        display_text_y = frame.shape[0] - 20
        if self.bar_length_px is not None:
            saved_text = f"Measured bar: {self.bar_length_px:.1f} px"
            # ... (bestaande code voor background en putText) ...
            text_size = cv2.getTextSize(saved_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (10, display_text_y - 25), (10 + text_size[0] + 10, display_text_y + 5), (0,0,0), -1)
            cv2.putText(frame, saved_text, (15, display_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            display_text_y -= 30 # Schuif volgende regel omhoog

        if self.pixels_per_meter is not None:
            calib_text = f"Current Calibration: {self.pixels_per_meter:.2f} px/m (for {self.ACTUAL_BAR_LENGTH_METERS}m bar)"
            # ... (bestaande code voor background en putText) ...
            text_size = cv2.getTextSize(calib_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0] # Iets kleinere tekst
            cv2.rectangle(frame, (10, display_text_y - 22), (10 + text_size[0] + 10, display_text_y + 5),(0,0,0),-1)
            cv2.putText(frame, calib_text, (15, display_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2) # Andere kleur
                
    def update_rep_state(self, current_y):
        """Update the rep state machine based on bar movement and position"""
        try:
            # Check if we need to wait more time in the current state for debouncing
            current_time = time.time()
            time_in_state = current_time - self.state_change_time
            
            # Store current y position if it's the first stable detection
            if self.start_y_position is None and self.movement_state == "STABLE":
                self.start_y_position = current_y
                print(f"Set start_y_position to {self.start_y_position}")
            
            # Update min/max positions for debugging
            if current_y < self.debug_info["min_y"]:
                self.debug_info["min_y"] = current_y
            if current_y > self.debug_info["max_y"]:
                self.debug_info["max_y"] = current_y
            
            # Debug output for troubleshooting
            print(f"Rep state: {self.rep_state}, Movement: {self.movement_state}, Start Y: {self.start_y_position}, Current Y: {current_y}")
            print(f"  Y range: min={self.debug_info['min_y']:.1f}, max={self.debug_info['max_y']:.1f}, current={current_y:.1f}")
            
            # Print stable time info as debugging
            if self.stable_detection_time is not None:
                stable_duration = current_time - self.stable_detection_time
                print(f"  Stable for {stable_duration:.2f}s (min={self.min_time_for_stable}s)")
            
            # Calculate current speed and update max speed if needed
            self.calculate_current_speed(current_y, current_time)
            
            # Detect inconsistent states and correct them
            self.detect_and_correct_state_inconsistencies()
                
            # Determine the potential next state based on current movement and state
            potential_state = self.rep_state  # Default to current state
            
            # In case we get stuck in an error state, force reset to READY
            if self.rep_state not in ["READY", "DESCENDING", "BOTTOM", "ASCENDING", "COMPLETED"]:
                potential_state = "READY"
                self.reset_rep_tracking()
                print("RECOVERY: Reset to READY state due to invalid state")
            
            if self.rep_state == "READY":
                # Check voor pending reset van een mislukte ASCENDING poging
                if self.pending_reset_after_failed_ascent:
                    print("Executing pending reset after failed ascent.")
                    self.reset_rep_tracking()
                    self.pending_reset_after_failed_ascent = False

                # Maak het eenvoudiger om een rep te starten: als er beweging naar beneden is, begin de rep
                if self.movement_state == "DOWNWARD":
                    # Als we nog geen startpositie hebben, gebruik de huidige positie
                    if self.start_y_position is None:
                        self.start_y_position = current_y
                        print(f"Auto-setting start_y_position to {self.start_y_position}")
                        
                    # Controleer of de beweging substantieel genoeg is
                    # We maken een initiële schatting - echte validatie volgt later
                    y_diff = abs(current_y - self.start_y_position)
                    if y_diff >= 5:  # Minimale beweging om ruis te vermijden
                        potential_state = "DESCENDING"
                        print("STATE TRANSITION: READY -> DESCENDING (detected downward movement)")
                        # Initialize rep stats
                        if self.current_rep_stats["start_time"] is None:
                            self.current_rep_stats["start_time"] = current_time
                            self.current_rep_stats["start_y"] = current_y # Dit is de initiële start Y
                            self.current_rep_stats["bottom_y"] = current_y # Initieel is bottom_y gelijk aan start_y
                            # Reset speed tracking
                            self.speed_buffer = []
                            self.last_y_for_speed = current_y
                            self.last_time_for_speed = current_time
                    
            elif self.rep_state == "DESCENDING":
                # Update max descent speed and phase_duration (bestaande code)
                if len(self.speed_buffer) > 0:
                    current_speed = abs(self.speed_buffer[-1])
                    if current_speed > self.current_rep_stats["max_descent_speed"]:
                        self.current_rep_stats["max_descent_speed"] = current_speed
                phase_duration = 0
                if self.current_rep_stats["start_time"] is not None:
                    phase_duration = current_time - self.current_rep_stats["start_time"]

                state_debug = "DESCENDING -> "

                # Blijf bottom_y updaten zolang we naar beneden gaan
                if self.movement_state == "DOWNWARD":
                    if self.current_rep_stats["bottom_y"] is None or current_y > self.current_rep_stats["bottom_y"]:
                        self.current_rep_stats["bottom_y"] = current_y
                    state_debug += "MOVING_DOWN "
                    if hasattr(self, 'stabilized_bar_angle'): # Voeg hoek toe als attribuut bestaat
                        self.current_rep_stats["descent_angles"].append(self.stabilized_bar_angle)

                elif self.movement_state == "STABLE":
                    state_debug += "STABLE_DETECTED "
                    if self.stable_detection_time is not None:
                        stable_duration = current_time - self.stable_detection_time
                        # Gebruik de geüpdatete bottom_y voor de afstandscheck
                        effective_descent = 0
                        if self.current_rep_stats["start_y"] is not None and self.current_rep_stats["bottom_y"] is not None:
                             effective_descent = abs(self.current_rep_stats["bottom_y"] - self.current_rep_stats["start_y"])
                        
                        state_debug += f"StableDur: {stable_duration:.2f}s, EffDescent: {effective_descent:.1f}px "

                        if stable_duration >= self.min_time_for_stable and effective_descent >= self.min_rep_distance * 0.7:
                            state_debug += "TO_BOTTOM_CONDITIONS_MET "
                            potential_state = "BOTTOM"
                            # bottom_y is al gezet, nu definitief maken voor rep stats en bottom_y_position
                            self.bottom_y_position = self.current_rep_stats["bottom_y"] 
                            self.current_rep_stats["descent_distance"] = effective_descent
                            self.current_rep_stats["bottom_time"] = current_time # Tijd van bereiken BOTTOM

                            if self.current_rep_stats["start_time"] is not None:
                                descent_time = self.current_rep_stats["bottom_time"] - self.current_rep_stats["start_time"]
                                if descent_time > 0:
                                    self.current_rep_stats["avg_descent_speed"] = self.current_rep_stats["descent_distance"] / descent_time
                            self.speed_buffer = []
                            self.consecutive_state_frames = self.state_frames_threshold
                            self.consecutive_upward_frames_in_bottom = 0 # Reset voor nieuwe BOTTOM state
                            self.upward_trend_frames = 0 # Reset ook deze voor de zekerheid

                        elif stable_duration >= self.min_time_for_stable and effective_descent < self.min_rep_distance * 0.7:
                            state_debug += f"STABLE_BUT_DIST_SMALL ({effective_descent:.1f}px < {self.min_rep_distance * 0.7:.1f}px) "
                            if current_time - self.stable_detection_time > 1.5: # Verhoogde timeout voor reset
                                state_debug += "TIMEOUT_RESET_TO_READY "
                                potential_state = "READY"
                                self.reset_rep_tracking()
                        else: # stable_duration < min_time_for_stable
                             state_debug += f"WAITING_STABLE_DURATION ({stable_duration:.2f}s < {self.min_time_for_stable}s) "
                    else: # self.stable_detection_time is None (onwaarschijnlijk hier, maar voor volledigheid)
                        state_debug += "STABLE_BUT_NO_STABLE_TIME_YET "
                
                elif self.movement_state == "UPWARD": # Vroegtijdige opwaartse beweging in DESCENDING
                    state_debug += "EARLY_UPWARD_IN_DESCENDING "
                    # Dit kan een bounce zijn. Als het significant is, overweeg resetten of direct naar BOTTOM forceren.
                    # Voor nu, laten we de STABLE detectie afwachten om te beslissen.
                    # Als deze UPWARD aanhoudt, zal de volgende state check het mogelijk opvangen.

                if state_debug != self.debug_info["last_state_debug"]:
                    print(f"DEBUG: {state_debug}")
                    self.debug_info["last_state_debug"] = state_debug

            elif self.rep_state == "BOTTOM":
                if self.current_rep_stats["bottom_time"] is not None:
                    self.current_rep_stats["pause_duration"] = current_time - self.current_rep_stats["bottom_time"]
                
                print(f"  BOTTOM: Pause {self.current_rep_stats['pause_duration']:.2f}s. Movement: {self.movement_state}, UpwardFrames: {self.consecutive_upward_frames_in_bottom}")

                min_pause_at_bottom = 0.1  # 100ms

                if self.movement_state == "UPWARD":
                    self.consecutive_upward_frames_in_bottom += 1
                    print(f"    BOTTOM Check: movement_state is UPWARD. consecutive_upward_frames_in_bottom = {self.consecutive_upward_frames_in_bottom}")
                else: # STABLE of DOWNWARD (bounce)
                    if self.consecutive_upward_frames_in_bottom > 0:
                         print(f"    BOTTOM Check: movement_state ({self.movement_state}) broke upward trend. Resetting consecutive_upward_frames_in_bottom.")
                    self.consecutive_upward_frames_in_bottom = 0
                
                required_consecutive_upward = 3 # Vereis 3 opeenvolgende frames met UPWARD movement_state
                if self.consecutive_upward_frames_in_bottom >= required_consecutive_upward and self.current_rep_stats["pause_duration"] >= min_pause_at_bottom:
                    potential_state = "ASCENDING"
                    print(f"STATE TRANSITION: BOTTOM -> ASCENDING (ConsecUpwardFrames: {self.consecutive_upward_frames_in_bottom}>={required_consecutive_upward}, Pause: {self.current_rep_stats['pause_duration']:.2f}s >= {min_pause_at_bottom}s)")
                    self.speed_buffer = []
                    self.last_y_for_speed = current_y
                    self.last_time_for_speed = current_time
                    self.consecutive_state_frames = self.state_frames_threshold
                    self.downward_frames_in_ascent = 0 
                    # Reset real-time power stats voor de nieuwe stijging
                    self.current_rep_stats["rt_instant_power"] = 0
                    self.current_rep_stats["rt_peak_power_this_ascent"] = 0
                elif time_in_state > 5.0: # Timeout
                    potential_state = "READY"
                    print(f"STATE TRANSITION: BOTTOM -> READY (timeout after {time_in_state:.2f}s)")
                    self.reset_rep_tracking()

            elif self.rep_state == "ASCENDING":
                self.upward_trend_frames = 0
                # self.consecutive_upward_frames_in_bottom is gereset.
                # self.downward_frames_in_ascent is gereset bij BOTTOM -> ASCENDING transitie.
                
                if hasattr(self, 'stabilized_bar_angle'):
                    self.current_rep_stats["ascent_angles"].append(self.stabilized_bar_angle)
                
                # Real-time power berekening voor display
                if self.speed_buffer and self.pixels_per_meter is not None and self.pixels_per_meter > 0 and self.total_weight > 0:
                    latest_speed_pxps = self.speed_buffer[-1] # Pak de meest recente instantane snelheid
                    if latest_speed_pxps < 0: # Alleen als opwaarts (snelheid is negatief)
                        latest_speed_mps = abs(latest_speed_pxps) / self.pixels_per_meter
                        force = self.total_weight * self.GRAVITY
                        instant_power = force * latest_speed_mps
                        self.current_rep_stats["rt_instant_power"] = instant_power
                        # Gebruik .get met een default van 0 voor het geval de key nog niet bestaat (zou niet mogen, maar safe)
                        if instant_power > self.current_rep_stats.get("rt_peak_power_this_ascent", 0):
                            self.current_rep_stats["rt_peak_power_this_ascent"] = instant_power
                            # DEBUG PRINT voor wanneer piekvermogen wordt bijgewerkt
                            print(f"  ** Peak Power (RT) Updated: {instant_power:.0f}W (Speed: {latest_speed_mps:.2f}m/s from {abs(latest_speed_pxps):.1f}px/s, Force: {force:.1f}N, Px/M: {self.pixels_per_meter:.1f}, Weight: {self.total_weight}kg) **")
                    else: # Niet opwaarts (of snelheid is 0), dus huidig instant vermogen is 0
                        self.current_rep_stats["rt_instant_power"] = 0
                else: # Geen kalibratie of gewicht, zet instant power ook op 0 voor display
                    self.current_rep_stats["rt_instant_power"] = 0

                # Update max_ascent_speed (voor de uiteindelijke rep statistiek)
                # Dit moet nog steeds de absolute snelheid gebruiken, maar alleen als het opwaarts is.
                if self.speed_buffer and self.speed_buffer[-1] < 0: # Check of laatste snelheid opwaarts is
                    current_raw_ascent_speed_pxps = abs(self.speed_buffer[-1])
                    if current_raw_ascent_speed_pxps > self.current_rep_stats["max_ascent_speed"]:
                        self.current_rep_stats["max_ascent_speed"] = current_raw_ascent_speed_pxps
                
                phase_duration = 0
                if self.current_rep_stats["bottom_time"] is not None:
                    phase_duration = current_time - self.current_rep_stats["bottom_time"]
                
                # CONDITIE 1: Normale voltooiing - STABLE nabij start_y
                if self.movement_state == "STABLE" and self.start_y_position is not None and self.bottom_y_position is not None:
                    self.downward_frames_in_ascent = 0 
                    vertical_distance_from_start = abs(current_y - self.start_y_position)
                    top_bottom_range = abs(self.start_y_position - self.bottom_y_position)
                    threshold_percentage = 0.35 
                    min_ascent_percentage = 0.65
                    if top_bottom_range > 0 and phase_duration >= self.min_phase_duration: 
                        threshold_pixels = top_bottom_range * threshold_percentage
                        current_ascent_distance = abs(current_y - self.bottom_y_position)
                        print(f"  ASCENDING->COMPLETED Check 1: Stable. DistFromStart: {vertical_distance_from_start:.1f}px (Threshold: {threshold_pixels:.1f}px). AscentDist: {current_ascent_distance:.1f}px (MinReq: {top_bottom_range * min_ascent_percentage:.1f}px)")
                        if vertical_distance_from_start <= threshold_pixels and current_ascent_distance >= top_bottom_range * min_ascent_percentage:
                            total_vertical_movement = self.current_rep_stats["descent_distance"] 
                            if total_vertical_movement >= self.min_rep_distance:
                                if not self.current_rep_stats.get("_rep_counted_this_cycle", False):
                                    print(f"STATE TRANSITION: ASCENDING -> COMPLETED (Cond 1: Reached top stable) - Counting Rep")
                                    self.current_rep_stats["end_time"] = current_time; self.current_rep_stats["end_y"] = current_y
                                    if self.current_rep_stats["bottom_y"] is not None: self.current_rep_stats["ascent_distance"] = abs(current_y - self.current_rep_stats["bottom_y"])
                                    if self.current_rep_stats["bottom_time"] is not None:
                                        ascent_time = current_time - self.current_rep_stats["bottom_time"] - self.current_rep_stats["pause_duration"]
                                        if ascent_time > 0: self.current_rep_stats["avg_ascent_speed"] = self.current_rep_stats["ascent_distance"] / ascent_time
                                    
                                    # Bereken Power en Hoek statistieken
                                    print(f"DEBUG POWER CALC: pixels_per_meter={self.pixels_per_meter}, total_weight={self.total_weight}") # DEBUG PRINT
                                    if self.pixels_per_meter is not None and self.pixels_per_meter > 0 and self.total_weight > 0:
                                        # Snelheden omrekenen naar m/s
                                        avg_ascent_speed_mps = (self.current_rep_stats["avg_ascent_speed"] / self.pixels_per_meter) if self.current_rep_stats["avg_ascent_speed"] is not None else 0
                                        # max_ascent_speed_mps = (self.current_rep_stats["max_ascent_speed"] / self.pixels_per_meter) if self.current_rep_stats["max_ascent_speed"] is not None else 0 # Niet meer nodig voor peak power
                                        force = self.total_weight * self.GRAVITY
                                        self.current_rep_stats["avg_power_ascent"] = force * avg_ascent_speed_mps
                                        # Gebruik de rt_peak_power_this_ascent als de definitieve peak_power_ascent
                                        self.current_rep_stats["peak_power_ascent"] = self.current_rep_stats.get("rt_peak_power_this_ascent", 0) 
                                    
                                    if self.current_rep_stats["descent_angles"]:
                                        angles = np.array(self.current_rep_stats["descent_angles"])
                                        self.current_rep_stats["avg_angle_descent"] = np.mean(angles)
                                        self.current_rep_stats["std_angle_descent"] = np.std(angles)
                                        self.current_rep_stats["max_abs_angle_descent"] = np.max(np.abs(angles))
                                    if self.current_rep_stats["ascent_angles"]:
                                        angles = np.array(self.current_rep_stats["ascent_angles"])
                                        self.current_rep_stats["avg_angle_ascent"] = np.mean(angles)
                                        self.current_rep_stats["std_angle_ascent"] = np.std(angles)
                                        self.current_rep_stats["max_abs_angle_ascent"] = np.max(np.abs(angles))

                                    # ... (bestaande code voor rep tellen, beep, vlag)
                                    print(f"STATE TRANSITION: ASCENDING -> COMPLETED (Cond 1: Reached top stable) - Counting Rep")
                                    self.rep_history.append(self.current_rep_stats.copy())
                                    self.rep_count += 1
                                    if hasattr(self, 'rep_complete_sound'): self.rep_complete_sound.play()
                                    print(f"REP COUNTED: {self.rep_count}. Stats: {self.current_rep_stats}")
                                    self.current_rep_stats["_rep_counted_this_cycle"] = True
                                potential_state = "COMPLETED"
                            else:
                                potential_state = "READY"
                                print(f"STATE CORRECTION: ASCENDING -> READY (Cond 1: Stable at top, but insufficient total rep movement: {total_vertical_movement}px)")
                                self.reset_rep_tracking() # Reset als rep ongeldig is
                
                # CONDITIE 2: Vangnet voor als stang neerwaarts beweegt in ASCENDING
                elif self.movement_state == "DOWNWARD":
                    self.downward_frames_in_ascent += 1
                    # ... (bestaande logica voor Conditie 2a en 2b) ...
                    print(f"  ASCENDING Check 2: Movement DOWNWARD detected. DownwardFramesInAscent: {self.downward_frames_in_ascent}")
                    ascent_so_far = 0
                    full_range_of_motion = 0
                    if self.bottom_y_position is not None and self.start_y_position is not None:
                        ascent_so_far = abs(self.bottom_y_position - current_y) 
                        full_range_of_motion = abs(self.bottom_y_position - self.start_y_position)
                    print(f"    Downward in ASC: AscentSoFar: {ascent_so_far:.1f}px, FullROM: {full_range_of_motion:.1f}px")

                    if full_range_of_motion > 0 and ascent_so_far >= full_range_of_motion * 0.85 and abs(current_y - self.start_y_position) < 20 and phase_duration >= self.min_phase_duration:
                        total_vertical_movement = self.current_rep_stats["descent_distance"]
                        if total_vertical_movement >= self.min_rep_distance:
                            if not self.current_rep_stats.get("_rep_counted_this_cycle", False):
                                print(f"STATE TRANSITION: ASCENDING -> COMPLETED (Cond 2a: Downward near top but sufficient ascent) - Counting Rep")
                                self.current_rep_stats["end_time"] = current_time; self.current_rep_stats["end_y"] = current_y
                                if self.current_rep_stats["bottom_y"] is not None: self.current_rep_stats["ascent_distance"] = ascent_so_far # Gebruik ascent_so_far hier
                                if self.current_rep_stats["bottom_time"] is not None:
                                    ascent_time = current_time - self.current_rep_stats["bottom_time"] - self.current_rep_stats["pause_duration"]
                                    if ascent_time > 0: self.current_rep_stats["avg_ascent_speed"] = self.current_rep_stats["ascent_distance"] / ascent_time

                                # Bereken Power en Hoek statistieken (zelfde als hierboven)
                                print(f"DEBUG POWER CALC (Cond 2a): pixels_per_meter={self.pixels_per_meter}, total_weight={self.total_weight}") # DEBUG PRINT
                                if self.pixels_per_meter is not None and self.pixels_per_meter > 0 and self.total_weight > 0:
                                    avg_ascent_speed_mps = (self.current_rep_stats["avg_ascent_speed"] / self.pixels_per_meter) if self.current_rep_stats["avg_ascent_speed"] is not None else 0
                                    # max_ascent_speed_mps = (self.current_rep_stats["max_ascent_speed"] / self.pixels_per_meter) if self.current_rep_stats["max_ascent_speed"] is not None else 0 # Niet meer nodig
                                    force = self.total_weight * self.GRAVITY
                                    self.current_rep_stats["avg_power_ascent"] = force * avg_ascent_speed_mps
                                    self.current_rep_stats["peak_power_ascent"] = self.current_rep_stats.get("rt_peak_power_this_ascent", 0)
                                if self.current_rep_stats["descent_angles"]:
                                    angles = np.array(self.current_rep_stats["descent_angles"])
                                    self.current_rep_stats["avg_angle_descent"] = np.mean(angles); self.current_rep_stats["std_angle_descent"] = np.std(angles); self.current_rep_stats["max_abs_angle_descent"] = np.max(np.abs(angles))
                                if self.current_rep_stats["ascent_angles"]:
                                    angles = np.array(self.current_rep_stats["ascent_angles"])
                                    self.current_rep_stats["avg_angle_ascent"] = np.mean(angles); self.current_rep_stats["std_angle_ascent"] = np.std(angles); self.current_rep_stats["max_abs_angle_ascent"] = np.max(np.abs(angles))
                                
                                # ... (bestaande code voor rep tellen, beep, vlag)
                                print(f"STATE TRANSITION: ASCENDING -> COMPLETED (Cond 2a: Downward near top but sufficient ascent) - Counting Rep")
                                self.rep_history.append(self.current_rep_stats.copy())
                                self.rep_count += 1
                                if hasattr(self, 'rep_complete_sound'): self.rep_complete_sound.play()
                                print(f"REP COUNTED: {self.rep_count}. Stats: {self.current_rep_stats}")
                                self.current_rep_stats["_rep_counted_this_cycle"] = True
                            potential_state = "COMPLETED"
                        else:
                            potential_state = "READY"
                            print(f"STATE CORRECTION: ASCENDING -> READY (Cond 2a: Downward near top, but insufficient total rep movement: {total_vertical_movement}px)")
                            self.reset_rep_tracking() # Reset als rep ongeldig is
                    elif self.downward_frames_in_ascent >= 3: 
                        potential_state = "READY"
                        self.pending_reset_after_failed_ascent = True 
                        print(f"STATE CORRECTION: ASCENDING -> READY (Cond 2b: Persistent downward movement ({self.downward_frames_in_ascent} frames) with insufficient ascent: {ascent_so_far:.1f}px) - RESET PENDING")
                    else:
                        print(f"    Downward in ASC: Tolerating for {self.downward_frames_in_ascent} frame(s). Waiting for recovery or more downward frames.")
                
                elif self.movement_state == "UPWARD":
                    if self.downward_frames_in_ascent > 0:
                        print(f"  ASCENDING: Movement back to UPWARD. Resetting downward_frames_in_ascent from {self.downward_frames_in_ascent}")
                    self.downward_frames_in_ascent = 0

                if hasattr(self, 'stabilized_bar_angle'): # Voeg hoek toe als attribuut bestaat
                    self.current_rep_stats["ascent_angles"].append(self.stabilized_bar_angle)

            elif self.rep_state == "COMPLETED":
                # Cooldown periode voor COMPLETED state. Rep is NU al geteld bij ASCENDING -> COMPLETED.
                self.downward_frames_in_ascent = 0 

                if time_in_state >= 0.8: 
                    print("STATE TRANSITION: COMPLETED -> READY (cooldown complete)")
                    next_start_y = self.current_rep_stats.get("end_y", self.start_y_position) 
                    self.reset_rep_tracking() 
                    self.start_y_position = next_start_y 
                    potential_state = "READY"
                    print(f"New rep cycle. Initial start_y_position set to: {self.start_y_position}")
                
                elif self.movement_state == "DOWNWARD": 
                    y_diff_from_end = 0
                    if self.current_rep_stats.get("end_y") is not None:
                        y_diff_from_end = abs(current_y - self.current_rep_stats["end_y"])
                    else: 
                        y_diff_from_end = abs(current_y - self.start_y_position) if self.start_y_position is not None else 10

                    if y_diff_from_end >= 10:  
                        print("STATE TRANSITION: COMPLETED -> DESCENDING (early new rep detected during cooldown)")
                        # De vorige rep is al geteld bij de ASCENDING -> COMPLETED overgang.
                        next_start_y = self.current_rep_stats.get("end_y", current_y) 
                        self.reset_rep_tracking()
                        self.start_y_position = next_start_y
                        self.current_rep_stats["start_time"] = current_time
                        self.current_rep_stats["start_y"] = self.start_y_position
                        self.current_rep_stats["bottom_y"] = self.start_y_position
                        self.speed_buffer = []
                        self.last_y_for_speed = current_y
                        self.last_time_for_speed = current_time
                        potential_state = "DESCENDING"
                        self.consecutive_state_frames = self.state_frames_threshold

            # Apply the state change with debouncing to avoid rapid switches
            if potential_state != self.rep_state:
                self.consecutive_state_frames += 1
                if self.consecutive_state_frames >= self.state_frames_threshold:
                    # Log the state change
                    self.rep_state_history.append((self.rep_state, potential_state, current_time))
                    
                    # Apply the new state
                    self.rep_state = potential_state
                    self.state_change_time = current_time
                    self.consecutive_state_frames = 0
                    
                    print(f"Rep state changed to {self.rep_state}")
                else:
                    print(f"Potential state change to {potential_state}, frame {self.consecutive_state_frames}/{self.state_frames_threshold}")
            else:
                self.consecutive_state_frames = 0
        except Exception as e:
            print(f"Error in update_rep_state: {e}")
            # Continue with normal detection even if rep tracking fails
            
    def reset_rep_tracking(self):
        """Reset all rep tracking variables to their initial state"""
        try:
            # Reset position tracking
            self.start_y_position = None
            self.bottom_y_position = None
            
            # Reset rep stats
            self.current_rep_stats = {
                "start_time": None,
                "bottom_time": None,
                "end_time": None,
                "pause_duration": 0,
                "descent_distance": 0, # in pixels
                "ascent_distance": 0,  # in pixels
                "max_descent_speed": 0, # in pixels/s
                "max_ascent_speed": 0,  # in pixels/s
                "avg_descent_speed": 0, # in pixels/s
                "avg_ascent_speed": 0,  # in pixels/s
                "start_y": None,
                "bottom_y": None,
                "end_y": None,
                "_rep_counted_this_cycle": False, 
                # Nieuwe statistieken
                "peak_power_ascent": None, # in Watts
                "avg_power_ascent": None,  # in Watts
                "descent_angles": [],
                "ascent_angles": [],
                "avg_angle_descent": None,
                "std_angle_descent": None,
                "max_abs_angle_descent": None,
                "avg_angle_ascent": None,
                "std_angle_ascent": None,
                "max_abs_angle_ascent": None
            }
            
            # Reset speed tracking
            self.speed_buffer = []
            self.last_y_for_speed = None
            self.last_time_for_speed = None
            self.stable_detection_time = None
            
            # Reset debug info minmax values
            self.debug_info["stable_at"] = None
            self.debug_info["transition_blocked_reason"] = None
            
            print("Reset all rep tracking variables")
        except Exception as e:
            print(f"Error in reset_rep_tracking: {e}")
            
    def detect_and_correct_state_inconsistencies(self):
        """Detect and correct inconsistent states in the state machine"""
        try:
            # Controleer of alle benodigde attributen bestaan
            if not hasattr(self, 'rep_state') or not hasattr(self, 'state_change_time'):
                return  # Skip als essentiële attributen nog niet bestaan
                
            # Debuggen van afstanden voor validatie
            if self.start_y_position is not None and self.bottom_y_position is not None:
                current_y_pos = self.prev_bar_y if self.prev_bar_y is not None else 0
                vertical_range = abs(self.start_y_position - self.bottom_y_position)
                current_vertical_pos = abs(current_y_pos - self.start_y_position)
                
                # Debug output voor validatie
                if self.rep_state in ["DESCENDING", "BOTTOM", "ASCENDING"]:
                    print(f"Vertical validation: Range={vertical_range}px, Current={current_vertical_pos}px, Min={self.min_rep_distance}px")
            
            # Detecteer of er inconsistenties zijn tussen de movement state en rep state
            if self.rep_state == "ASCENDING" and self.movement_state == "DOWNWARD":
                # Als we in ASCENDING zijn maar al naar beneden bewegen, is er iets mis
                print(f"INCONSISTENCY DETECTED: Rep state is {self.rep_state} but movement is {self.movement_state}")
                print("Auto-correcting state in next update...")
                # De correctie zal in de update_rep_state functie worden toegepast
            
            # Controleer of we te lang in dezelfde state blijven (mogelijk vastgelopen)
            current_time = time.time()
            max_time_in_state = 10.0  # Maximum tijd in seconden voor elke state
            
            if current_time - self.state_change_time > max_time_in_state:
                print(f"WARNING: Stuck in state {self.rep_state} for over {max_time_in_state} seconds")
                if self.rep_state != "READY":
                    print(f"Auto-resetting to READY state due to timeout")
                    self.rep_state = "READY"
                    self.state_change_time = current_time
                    # Reset tracking variables
                    self.reset_rep_tracking()
                    
        except Exception as e:
            print(f"Error in detect_and_correct_state_inconsistencies: {e}")

    def calculate_current_speed(self, current_y, current_time):
        """Calculate the current vertical speed of the bar"""
        try:
            if self.last_y_for_speed is None or self.last_time_for_speed is None:
                self.last_y_for_speed = current_y
                self.last_time_for_speed = current_time
                return
                
            # Calculate time difference
            time_diff = current_time - self.last_time_for_speed
            if time_diff < 0.001:  # Avoid division by zero or very small values
                return
                
            # Calculate distance and speed (negative is upward)
            y_diff = current_y - self.last_y_for_speed
            current_speed = y_diff / time_diff  # pixels per second
            
            # Add to speed buffer
            self.speed_buffer.append(current_speed)
            if len(self.speed_buffer) > self.speed_buffer_size:
                self.speed_buffer.pop(0)
                
            # Update last values
            self.last_y_for_speed = current_y
            self.last_time_for_speed = current_time
        except Exception as e:
            print(f"Error in calculate_current_speed: {e}")
            # Continue with normal detection even if speed calculation fails
        
    def get_current_speed(self):
        """Get the current smoothed speed (average of recent values)"""
        try:
            if not self.speed_buffer:
                return 0
            return sum(self.speed_buffer) / len(self.speed_buffer)
        except Exception as e:
            print(f"Error in get_current_speed: {e}")
            return 0

    def draw_rep_statistics(self, frame):
        """Draw statistics for the current rep"""
        try:
            if not self.roi_defined:
                return  # Don't draw stats if ROI is not defined
                
            # Controleer of rep_state bestaat voordat we het proberen te gebruiken
            if not hasattr(self, 'rep_state'):
                return  # Skip als rep_state nog niet bestaat
                
            # Set up drawing parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255)
            
            # Position at bottom right
            panel_width = 300
            panel_height = 180  # Increased height for more stats
            x = frame.shape[1] - panel_width - 10
            y = frame.shape[0] - panel_height - 10
            
            # Draw panel background
            cv2.rectangle(frame, 
                         (x, y), 
                         (x + panel_width, y + panel_height), 
                         bg_color, 
                         -1)
            cv2.rectangle(frame, 
                         (x, y), 
                         (x + panel_width, y + panel_height), 
                         (100, 100, 100), 
                         1)
                         
            # Panel title
            title = "REP STATISTICS"
            title_size = cv2.getTextSize(title, font, font_scale+0.2, thickness+1)[0]
            title_x = x + (panel_width - title_size[0]) // 2
            cv2.putText(frame, title, (title_x, y+20), font, font_scale+0.2, (0, 255, 255), thickness+1)
            
            # Line spacing
            line_height = 20
            start_y = y + 40
            
            # Format pause duration nicely (only if we have bottom_time)
            pause_str = "N/A"
            if self.current_rep_stats["bottom_time"] is not None:
                if self.rep_state == "BOTTOM":
                    # Update with current value if we're at the bottom
                    current_time = time.time()
                    pause_duration = current_time - self.current_rep_stats["bottom_time"]
                    pause_str = f"{pause_duration:.2f}s"
                else:
                    pause_str = f"{self.current_rep_stats['pause_duration']:.2f}s"
                    
            # Calculate current descent/ascent distances
            descent_distance = self.current_rep_stats["descent_distance"]
            ascent_distance = self.current_rep_stats["ascent_distance"]
            
            # Current speed (real-time)
            current_speed = self.get_current_speed()
            speed_direction = "UP" if current_speed < 0 else "DOWN" if current_speed > 0 else "STABLE"
            current_speed_str = f"{abs(current_speed):.2f} px/s {speed_direction}"
            
            # Format max and average speeds
            max_descent_speed = self.current_rep_stats["max_descent_speed"]
            max_ascent_speed = self.current_rep_stats["max_ascent_speed"]
            
            # Draw statistics
            cv2.putText(frame, f"Rep State: {self.rep_state}", 
                       (x+10, start_y), font, font_scale, text_color, thickness)
            cv2.putText(frame, f"Current Speed: {current_speed_str}", 
                       (x+10, start_y+line_height), font, font_scale, text_color, thickness)
            cv2.putText(frame, f"Descent: {descent_distance:.1f} px", 
                       (x+10, start_y+line_height*2), font, font_scale, text_color, thickness)
            cv2.putText(frame, f"Ascent: {ascent_distance:.1f} px", 
                       (x+10, start_y+line_height*3), font, font_scale, text_color, thickness)
            cv2.putText(frame, f"Bottom pause: {pause_str}", 
                       (x+10, start_y+line_height*4), font, font_scale, text_color, thickness)
            cv2.putText(frame, f"Max Descent Speed: {max_descent_speed:.2f} px/s", 
                       (x+10, start_y+line_height*5), font, font_scale, text_color, thickness)
            cv2.putText(frame, f"Max Ascent Speed: {max_ascent_speed:.2f} px/s", 
                       (x+10, start_y+line_height*6), font, font_scale, text_color, thickness)
            
            # Real-time peak power display
            rt_peak_power = self.current_rep_stats.get("rt_peak_power_this_ascent", 0)
            cv2.putText(frame, f"Peak Power (RT): {rt_peak_power:.0f} W", 
                       (x+10, start_y+line_height*7), font, font_scale, (50, 200, 255), thickness) # Opvallende kleur
        except Exception as e:
            print(f"Error in draw_rep_statistics: {e}")
            # Continue even if statistics drawing fails

    def reset_all(self):
        """Reset all tracking parameters and state machine when video restarts"""
        try:
            print("Resetting all parameters due to video restart")
            
            # Reset rep counter
            self.rep_count = 0
            
            # Reset tracking
            self.reset_rep_tracking()
            
            # Reset bar detection variables
            self.valid_detection = False
            self.prev_bar_y = None
            self.movement_state = "UNKNOWN"
            self.y_positions = []
            self.stable_frames = 0
            self.stable_detection_time = None
            self.bar_width_history = []
            self.bar_angle_history = []
            
            # Reset state machine
            self.rep_state = "READY"
            self.state_change_time = time.time()
            self.consecutive_state_frames = 0
            self.rep_state_history = []
            
            # Reset debug info
            self.debug_info = {
                "min_y": float('inf'),
                "max_y": float('-inf'),
                "stable_at": None,
                "transition_blocked_reason": None,
                "last_state_debug": None
            }
            
            # Set reset feedback
            self.reset_message = "PARAMETERS RESET - READY TO START"
            self.reset_time = time.time()
            
            print("All parameters reset successfully")
        except Exception as e:
            print(f"Error in reset_all: {e}")
            
    def draw_reset_message(self, frame):
        """Draw a reset message on the frame if active"""
        if self.reset_message is not None and self.reset_time is not None:
            # Check if message should still be shown
            elapsed = time.time() - self.reset_time
            if elapsed > self.reset_message_duration:
                self.reset_message = None
                self.reset_time = None
                return
                
            # Calculate fade effect (1.0 to 0.0)
            opacity = 1.0 - (elapsed / self.reset_message_duration)
            
            # Draw background overlay
            overlay = frame.copy()
            h, w = frame.shape[:2]
            bg_color = (0, 0, 0)
            
            # Draw semi-transparent background
            cv2.rectangle(overlay, (0, h//2 - 40), (w, h//2 + 40), bg_color, -1)
            cv2.addWeighted(overlay, 0.5 * opacity, frame, 1 - 0.5 * opacity, 0, frame)
            
            # Draw message
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            text_size = cv2.getTextSize(self.reset_message, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2 + 10
            
            # Draw text with slight glow effect
            text_color = (0, 255, 255)  # Cyan color
            cv2.putText(frame, self.reset_message, (text_x, text_y), font, font_scale, 
                        (0, 100, 100), thickness + 2)  # Darker outline
            cv2.putText(frame, self.reset_message, (text_x, text_y), font, font_scale, 
                        text_color, thickness)
        
    def run(self):
        try:
            last_frame_time = time.time()
            
            while self.cap.isOpened():
                # Only read a new frame if not paused
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        if self.mode == 'debug':
                            print("End of video reached - restarting")
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.reset_all()  # Reset all parameters when video restarts
                            continue
                        break
                    # Store the current frame for use when paused
                    self.current_frame = frame.copy()
                else:
                    # Use the stored frame when paused
                    frame = self.current_frame.copy()
                
                # Process frame
                if self.mode == 'live' and not self.paused:
                    frame = cv2.flip(frame, 1)
                
                # Draw mode indicator and weight display
                self.draw_mode_indicator(frame)
                self.draw_weight_display(frame)
                
                # Draw navigation feedback
                self.draw_navigation_feedback(frame)
                
                # Draw reset message if active
                self.draw_reset_message(frame)
                
                # Draw pause indicator if paused
                if self.paused:
                    pause_text = "PAUSED"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size = cv2.getTextSize(pause_text, font, 1.5, 3)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = 50
                    cv2.putText(frame, pause_text, (text_x, text_y), font, 1.5, (0, 0, 255), 3)
                
                # Draw HSV controls info (not in measuring mode) - now empty
                if not self.measuring_mode:
                    self.draw_hsv_controls_info(frame)
                
                # Draw bar measurement interface if in measuring mode
                if self.measuring_mode:
                    self.draw_measurement_line(frame)
                else:
                    # Only show ROI selection when not in measuring mode
                    # Add ROI instructions if not defined
                    if not self.roi_defined and not self.roi_selecting:
                        cv2.putText(frame, "Click and drag to select bar movement area", 
                                  (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 255, 255), 2)
                    
                    # Draw the ROI if it's being selected or already defined
                    if self.roi_selecting:
                        cv2.rectangle(frame, self.roi_start_point, self.roi_end_point, (0, 255, 255), 2)
                        cv2.putText(frame, "Drawing ROI...", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    elif self.roi_defined:
                        cv2.rectangle(frame, self.roi_start_point, self.roi_end_point, (0, 165, 255), 2)
                
                # Process the frame for bar detection (only when not in measuring mode)
                if not self.measuring_mode and self.roi_defined and not self.paused:
                    self.calculate_bar_metrics(frame)
                    
                    # Draw rep statistics if ROI is defined
                    if self.roi_defined:
                        self.draw_rep_statistics(frame)
                
                # Display the frame
                cv2.imshow("Main", frame)
                
                # Calculate frame timing for next frame
                current_time = time.time()
                frame_interval = 1.0 / self.fps
                if self.mode == 'debug':
                    # Adjust frame interval based on speed factor
                    frame_interval *= self.speed_factor
                
                # Calculate time to wait - use a small value when paused to keep UI responsive
                if not self.paused:
                    elapsed = current_time - last_frame_time
                    wait_time = max(1, int((frame_interval - elapsed) * 1000))
                else:
                    wait_time = 30  # When paused, check for keys more frequently (30ms)
                
                # Handle key presses and mouse events
                key = cv2.waitKey(wait_time) & 0xFF
                
                # Update last frame time if not paused
                if not self.paused:
                    last_frame_time = current_time
                
                # Check for arrow keys (left=81, right=83 on some platforms)
                full_key = cv2.waitKey(1)  # Just to check if there's an extended key
                is_left_arrow = key == 2 or full_key & 0xFF00 == 0x5000 and full_key & 0x00FF == 81
                is_right_arrow = key == 3 or full_key & 0xFF00 == 0x5000 and full_key & 0x00FF == 83
                
                if key == ord('q'):
                    break
                elif key == ord('m'):  # Switch mode
                    self.switch_mode()
                elif key == ord('s') and self.mode == 'live' and not self.is_tracking:
                    self.start_countdown()
                elif key == ord('+') or key == ord('='):
                    self.adjust_weight(10)
                elif key == ord('-') or key == ord('_'):
                    self.adjust_weight(-10)
                elif key == ord('p'):
                    self.adjust_weight(2.5)
                elif key == ord('o'):
                    self.adjust_weight(-2.5)
                elif key == ord('['):  # Decrease speed
                    self.speed_factor = min(self.max_speed_factor, self.speed_factor * 1.2)
                elif key == ord(']'):  # Increase speed
                    self.speed_factor = max(1.0, self.speed_factor / 1.2)
                elif key == ord(' ') and self.mode == 'debug':
                    # Toggle pause state instead of blocking with waitKey(0)
                    self.paused = not self.paused
                    if self.paused:
                        print("Video paused - UI is still responsive")
                    else:
                        print("Video resumed")
                elif key == ord('r'):  # Reset ROI
                    self.roi_defined = False
                    self.roi_selecting = False
                    if os.path.exists(self.roi_file):
                        try:
                            os.remove(self.roi_file)
                            print(f"ROI settings deleted. Draw a new ROI.")
                        except Exception as e:
                            print(f"Error deleting ROI file: {e}")
                elif key == ord('R'):  # Hoofdletter R voor volledige reset
                    self.reset_all()
                    print("All parameters reset manually - rep count and state machine are back to default")
                elif key == ord('b'):  # Toggle bar measurement mode
                    self.measuring_mode = not self.measuring_mode
                    if self.measuring_mode:
                        print("Entered bar measurement mode - draw a line along the bar")
                        # Reset measurement points when entering the mode
                        self.measure_start_point = None
                        self.measure_end_point = None
                    else:
                        print("Exited bar measurement mode")
                elif key == ord('h'):  # Toggle HSV Controls window
                    self.show_hsv_controls = not self.show_hsv_controls
                    if self.show_hsv_controls:
                        cv2.namedWindow("HSV Controls")
                        self.create_hsv_trackbars()
                        print("HSV Controls window opened")
                    else:
                        cv2.destroyWindow("HSV Controls")
                        print("HSV Controls window closed")
                elif key == ord('d'):  # Toggle Bar Detection window
                    self.show_bar_detection = not self.show_bar_detection
                    if self.show_bar_detection:
                        cv2.namedWindow("Bar Detection", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Bar Detection", 400, 300)
                        print("Bar Detection window opened")
                    else:
                        cv2.destroyWindow("Bar Detection")
                        print("Bar Detection window closed")
                elif is_left_arrow:
                    if self.mode == 'debug':
                        # Calculate frames to skip (0.5 seconds)
                        frames_to_skip = int(self.fps * 0.5)
                        # Get current position
                        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        # Set new position
                        new_pos = max(0, current_pos - frames_to_skip)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                        print(f"Skipped backward 0.5s: Frame {current_pos} → {new_pos}")
                        # Show navigation feedback
                        self.show_navigation_feedback('backward')
                elif is_right_arrow:
                    if self.mode == 'debug':
                        # Calculate frames to skip (0.5 seconds)
                        frames_to_skip = int(self.fps * 0.5)
                        # Get current position
                        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        # Get total frame count
                        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        # Set new position
                        new_pos = min(total_frames - 1, current_pos + frames_to_skip)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                        print(f"Skipped forward 0.5s: Frame {current_pos} → {new_pos}")
                        # Show navigation feedback
                        self.show_navigation_feedback('forward')
                
                # HSV keyboard controls
                elif key == ord('1'):  # Decrease H min
                    self.adjust_hsv_value(0, 'lower', -1)
                elif key == ord('!'):  # Increase H min
                    self.adjust_hsv_value(0, 'lower', 1)
                elif key == ord('2'):  # Decrease H max
                    self.adjust_hsv_value(0, 'upper', -1)
                elif key == ord('@'):  # Increase H max
                    self.adjust_hsv_value(0, 'upper', 1)
                elif key == ord('3'):  # Decrease S min
                    self.adjust_hsv_value(1, 'lower', -1)
                elif key == ord('#'):  # Increase S min
                    self.adjust_hsv_value(1, 'lower', 1)
                elif key == ord('4'):  # Decrease S max
                    self.adjust_hsv_value(1, 'upper', -1)
                elif key == ord('$'):  # Increase S max
                    self.adjust_hsv_value(1, 'upper', 1)
                elif key == ord('5'):  # Decrease V min
                    self.adjust_hsv_value(2, 'lower', -1)
                elif key == ord('%'):  # Increase V min
                    self.adjust_hsv_value(2, 'lower', 1)
                elif key == ord('6'):  # Decrease V max
                    self.adjust_hsv_value(2, 'upper', -1)
                elif key == ord('^'):  # Increase V max
                    self.adjust_hsv_value(2, 'upper', 1)
                    
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup all resources"""
        # Sla de rep history op voordat we afsluiten
        self.save_session_history()

        pygame.mixer.stop()
        pygame.mixer.quit()
        self.cleanup_capture()
        # Destroy all windows
        for name in self.debug_windows:
            cv2.destroyWindow(name)
        cv2.destroyAllWindows()

    def save_session_history(self, filename="session_stats.json"):
        """Slaat de self.rep_history op naar een JSON bestand."""
        if not self.rep_history:
            print("No rep history to save.")
            return

        try:
            history_to_save = []
            for rep_stats in self.rep_history:
                serializable_stats = rep_stats.copy() 
                
                # Verwijder de grote hoek-arrays en real-time power velden
                serializable_stats.pop("descent_angles", None)
                serializable_stats.pop("ascent_angles", None)
                serializable_stats.pop("rt_instant_power", None)
                serializable_stats.pop("rt_peak_power_this_ascent", None)
                
                # Converteer timestamps naar leesbaar formaat
                for time_key in ["start_time", "bottom_time", "end_time"]:
                    if serializable_stats.get(time_key) is not None:
                        ts = serializable_stats[time_key]
                        milliseconds = int((ts - int(ts)) * 1000)
                        serializable_stats[time_key] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) + f".{milliseconds:03d}"
                
                # Converteer overige numpy types naar Python floats/lists
                for key, value in serializable_stats.items():
                    if isinstance(value, np.ndarray):
                        serializable_stats[key] = value.tolist()
                    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                        serializable_stats[key] = float(value)
                    # Rond floats af op bijvoorbeeld 2 decimalen voor betere leesbaarheid, behalve hoeken
                    if isinstance(value, float) and "angle" not in key:
                        serializable_stats[key] = round(value, 2)
                    elif isinstance(value, float) and "angle" in key:
                        serializable_stats[key] = round(value, 1) # Hoeken met 1 decimaal
                        
                history_to_save.append(serializable_stats)

            with open(filename, 'w') as f:
                json.dump(history_to_save, f, indent=4)
            print(f"Session history saved to {filename}")
        except Exception as e:
            print(f"Error saving session history: {e}")

def generate_beep(duration, frequency, amplitude=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal.astype(np.float32)

def create_countdown():
    sample_rate = 44100
    
    # Generate three beeps with decreasing pitch
    beep1 = generate_beep(0.2, 880)  # A5 note
    beep2 = generate_beep(0.2, 784)  # G5 note
    beep3 = generate_beep(0.3, 660)  # E5 note
    
    # Add silence between beeps
    silence = np.zeros(int(sample_rate * 0.8))
    
    # Combine all sounds
    countdown = np.concatenate([
        beep1, silence,
        beep2, silence,
        beep3
    ])
    
    # Normalize and convert to 16-bit integer
    countdown = np.int16(countdown * 32767)
    
    # Save the file
    wavfile.write('countdown.wav', sample_rate, countdown)

if __name__ == "__main__":
    # Create beep.wav if it doesn't exist
    if not os.path.exists('beep.wav'):
        print("Generating beep.wav file...")
        sample_rate = 44100
        duration = 0.1  # 100ms beep
        frequency = 880  # A5 note
        amplitude = 0.5
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
        signal = np.int16(signal * 32767)
        wavfile.write('beep.wav', sample_rate, signal)
        print("beep.wav file created successfully!")
    
    # Start in debug mode by default
    tracker = BenchPressTracker(mode='debug')
    
    # Create countdown.wav if needed
    create_countdown()
    
    tracker.run()

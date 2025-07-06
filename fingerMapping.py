import cv2
import random
import time
import numpy as np
import mediapipe as mp

class RockPaperScissorsGame:
    def __init__(self):
        # Initialize MediaPipe hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only detect one hand for simplicity
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Game state variables
        self.cap = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.countdown = 3
        self.start_time = None
        self.start_countdown = False
        self.move_captured = False
        self.indicator_counter = 0
        self.game_result = None
        self.computer_choice = None
        self.user_choice = None
        
        # Statistics
        self.stats = {
            'games': 0,
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'moves': {'rock': 0, 'paper': 0, 'scissors': 0}
        }
    
    def initialize_camera(self):
        """Initialize the webcam"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
        return True
    
    def count_raised_fingers(self, hand_landmarks):
        """
        Count the number of raised fingers.
        - Thumb is considered raised if it's more to the right (for right hand) or left (for left hand) than the base
        - Other fingers are considered raised if their tip is higher than their middle joint
        """
        finger_tips = [4, 8, 12, 16, 20]  # Landmark indices for fingertips
        finger_bases = [2, 5, 9, 13, 17]  # Landmark indices for finger bases
        
        # Check if we have a left or right hand
        # If the thumb is on the right side of the pinky, it's a left hand
        is_left_hand = hand_landmarks.landmark[4].x > hand_landmarks.landmark[20].x
        
        raised_fingers = 0
        
        # Special case for thumb
        if is_left_hand:
            # Left hand - thumb is up if it's to the left of its base
            if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_bases[0]].x:
                raised_fingers += 1
        else:
            # Right hand - thumb is up if it's to the right of its base
            if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_bases[0]].x:
                raised_fingers += 1
        
        # For other fingers - they're up if the tip is higher than the PIP joint (middle joint)
        for i in range(1, 5):  # index, middle, ring, pinky
            pip_joint = finger_bases[i] + 2  # PIP joint is 2 landmarks after the base
            if hand_landmarks.landmark[finger_tips[i]].y < hand_landmarks.landmark[pip_joint].y:
                raised_fingers += 1
        
        return raised_fingers
    
    def get_gesture_from_fingers(self, num_fingers):
        """Convert finger count to rock, paper, scissors gesture"""
        if num_fingers == 0:
            return 'rock'
        elif num_fingers in [2, 3]:  # Allow both 2 and 3 fingers for scissors
            return 'scissors'
        elif num_fingers >= 4:  # 4 or 5 fingers for paper
            return 'paper'
        else:  # 1 finger is ambiguous, could map to something or ask for clearer gesture
            return 'unknown'
    
    def get_computer_choice(self):
        """Generate random computer choice"""
        return random.choice(['rock', 'paper', 'scissors'])
    
    def determine_winner(self, user_choice, computer_choice):
        """Determine the winner based on rock-paper-scissors rules"""
        if user_choice == computer_choice:
            return "Tie!"
        
        if (user_choice == 'rock' and computer_choice == 'scissors') or \
           (user_choice == 'paper' and computer_choice == 'rock') or \
           (user_choice == 'scissors' and computer_choice == 'paper'):
            return "You win!"
        else:
            return "Computer wins!"
    
    def display_stats(self, frame):
        """Display game statistics on the frame"""
        h, w = frame.shape[:2]
        stats_text = f"Games: {self.stats['games']} | Wins: {self.stats['wins']} | Losses: {self.stats['losses']} | Ties: {self.stats['ties']}"
        cv2.putText(frame, stats_text, (10, h-10), self.font, 0.6, (0, 255, 255), 2)
        
        # Display move history
        moves_text = f"Your moves: Rock: {self.stats['moves']['rock']} | Paper: {self.stats['moves']['paper']} | Scissors: {self.stats['moves']['scissors']}"
        cv2.putText(frame, moves_text, (10, h-40), self.font, 0.6, (0, 255, 255), 2)
    
    def display_finger_guide(self, frame):
        """Display guide for finger gestures"""
        h, w = frame.shape[:2]
        
        # Create a semi-transparent box for the guide
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-300, 10), (w-10, 130), (0, 0, 0), -1)
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add text
        cv2.putText(frame, "Gesture Guide:", (w-290, 30), self.font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- Fist (0 fingers) = Rock", (w-290, 60), self.font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- 2-3 fingers = Scissors", (w-290, 90), self.font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- 4-5 fingers = Paper", (w-290, 120), self.font, 0.6, (255, 255, 255), 1)
    
    def process_frame(self, frame):
        """Process a single frame for hand detection and finger counting"""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find hands
        results = self.hands.process(rgb_frame)
        
        # Variables to return
        frame_with_landmarks = frame.copy()
        num_fingers = None
        gesture = "No hand detected"
        
        # Check if any hands were found
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks
                self.mp_drawing.draw_landmarks(
                    frame_with_landmarks,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Count raised fingers
                num_fingers = self.count_raised_fingers(hand_landmarks)
                
                # Get gesture from finger count
                gesture = self.get_gesture_from_fingers(num_fingers)
                
                # Display finger count and gesture
                cv2.putText(frame_with_landmarks, f"Fingers: {num_fingers}", (10, 30), 
                           self.font, 1, (0, 255, 0), 2)
                cv2.putText(frame_with_landmarks, f"Gesture: {gesture}", (10, 70), 
                           self.font, 1, (0, 255, 0), 2)
        
        return frame_with_landmarks, num_fingers, gesture
    
    def run_game(self):
        """Main game loop"""
        # Initialize camera
        if not self.initialize_camera():
            return
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                # Flip frame horizontally for a more intuitive mirror view
                frame = cv2.flip(frame, 1)
                
                # Process the frame
                display_frame, num_fingers, current_gesture = self.process_frame(frame)
                
                # Add finger guide
                self.display_finger_guide(display_frame)
                
                # Game state handling
                if not self.start_countdown:
                    # Ready state, waiting for user to start
                    instructions = "Press SPACE to start a new game"
                    cv2.putText(display_frame, instructions, (50, 120), self.font, 1, (0, 255, 0), 2)
                    
                    # Display game stats
                    self.display_stats(display_frame)
                else:
                    # Game in progress
                    elapsed = int(time.time() - self.start_time)
                    
                    if elapsed < self.countdown:
                        # Countdown phase
                        text = str(self.countdown - elapsed)
                        (w, h), _ = cv2.getTextSize(text, self.font, 4, 4)
                        x = (display_frame.shape[1] - w) // 2
                        y = (display_frame.shape[0] + h) // 2
                        cv2.putText(display_frame, text, (x, y), self.font, 4, (0, 0, 255), 4)
                        
                        # Additional instruction
                        instruction = "Get ready to show your move!"
                        cv2.putText(display_frame, instruction, (50, 50), self.font, 1, (255, 255, 0), 2)
                    
                    elif elapsed == self.countdown and not self.move_captured:
                        # Capture phase
                        self.move_captured = True
                        self.indicator_counter = 20  # Flash for 20 frames
                        
                        # Computer makes choice
                        self.computer_choice = self.get_computer_choice()
                        
                        # Use current gesture as player's choice
                        if current_gesture != "unknown" and current_gesture != "No hand detected":
                            self.user_choice = current_gesture
                            self.stats['moves'][self.user_choice] += 1
                            self.game_result = self.determine_winner(self.user_choice, self.computer_choice)
                            
                            # Update statistics
                            self.stats['games'] += 1
                            if "win" in self.game_result.lower():
                                self.stats['wins'] += 1
                            elif "tie" in self.game_result.lower():
                                self.stats['ties'] += 1
                            else:
                                self.stats['losses'] += 1
                        else:
                            self.user_choice = "unclear"
                            self.game_result = "Couldn't detect your gesture clearly"
                        
                        print(f"Computer: {self.computer_choice}, You: {self.user_choice}, Result: {self.game_result}")
                    
                    elif self.move_captured:
                        # Results phase
                        if self.indicator_counter > 0:
                            # Flash border effect
                            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1]-1, display_frame.shape[0]-1), 
                                         (0, 0, 255), 5)
                            self.indicator_counter -= 1
                        
                        # Display game results
                        y_pos = 120
                        line_height = 50
                        
                        cv2.putText(display_frame, f"Computer chose: {self.computer_choice}", 
                                   (50, y_pos), self.font, 1, (255, 0, 0), 2)
                        y_pos += line_height
                        
                        if self.user_choice != "unclear":
                            cv2.putText(display_frame, f"You chose: {self.user_choice}", 
                                       (50, y_pos), self.font, 1, (0, 255, 0), 2)
                            y_pos += line_height
                            
                            cv2.putText(display_frame, f"Result: {self.game_result}", 
                                       (50, y_pos), self.font, 1.5, (0, 0, 255), 3)
                        else:
                            cv2.putText(display_frame, "Your gesture wasn't clear enough", 
                                       (50, y_pos), self.font, 1, (0, 0, 255), 2)
                        
                        # Instructions for next game
                        cv2.putText(display_frame, "Press SPACE to play again", 
                                   (50, display_frame.shape[0] - 50), self.font, 1, (255, 255, 0), 2)
                
                # Display the frame
                cv2.imshow("Rock Paper Scissors Game", display_frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # Quit the game
                    break
                elif key == ord(' '):
                    # Start/restart game with spacebar
                    if not self.start_countdown or self.move_captured:
                        self.start_time = time.time()
                        self.start_countdown = True
                        self.move_captured = False
                        self.game_result = None
                        self.user_choice = None
                        self.computer_choice = None
        
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == "__main__":
    game = RockPaperScissorsGame()
    game.run_game()
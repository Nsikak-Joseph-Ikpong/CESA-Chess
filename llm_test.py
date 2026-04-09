from google import genai
from google.genai import types
import requests
import time
import signal
import sys
from requests.auth import HTTPDigestAuth
import math
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
ROBOT_IP = os.getenv("ROBOT_IP", "127.0.0.1")
USERNAME = os.getenv("ROBOT_USERNAME", "Default User")
PASSWORD = os.getenv("ROBOT_PASSWORD", "robotics")
BASE_URL = f"http://{ROBOT_IP}"

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if API key is loaded
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables!")

# ROBOT CONFIGURATION - Set which arm you're using
ROBOT_ARM = "LEFT"  # Change to "RIGHT" if using right arm
MECHUNIT = "ROB_L" if ROBOT_ARM == "LEFT" else "ROB_R"
TASK_NAME = "T_ROB_L" if ROBOT_ARM == "LEFT" else "T_ROB_R"
MODULE_NAME = "Module1"

# Workspace limits (adjust based on your robot's actual workspace)
WORKSPACE_LIMITS = {
    'x_min': -120,
    'x_max': 120,
    'y_min': -490,
    'y_max': -140,
    'z_min': 100,
    'z_max': 270,
}

# -----------------------------
# Robot HTTP Session
# -----------------------------
SESSION = requests.Session()
SESSION.auth = HTTPDigestAuth(USERNAME, PASSWORD)
HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


def post(endpoint, data=None):
    r = SESSION.post(BASE_URL + endpoint, headers=HEADERS, data=data)
    if not r.ok:
        print(f"Error response: {r.status_code}")
        print(f"Response text: {r.text}")
    r.raise_for_status()
    return r


def get(endpoint):
    r = SESSION.get(BASE_URL + endpoint)
    if not r.ok:
        print(f"Error response: {r.status_code}")
        print(f"Response text: {r.text}")
    r.raise_for_status()
    return r


# -----------------------------
# Quaternion Math
# -----------------------------
def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return (w, x, y, z)


def axis_angle_to_quaternion(axis, angle_deg):
    """Convert axis-angle to quaternion. Angle in degrees."""
    angle_rad = math.radians(angle_deg)
    half_angle = angle_rad / 2
    s = math.sin(half_angle)

    w = math.cos(half_angle)
    x = axis[0] * s
    y = axis[1] * s
    z = axis[2] * s

    return (w, x, y, z)


def rotate_quaternion(q, axis, angle_deg):
    """Apply a rotation to a quaternion"""
    rotation = axis_angle_to_quaternion(axis, angle_deg)
    result = quaternion_multiply(rotation, q)
    return result


# -----------------------------
# RAPID control
# -----------------------------
def start_rapid():
    """Start RAPID execution on robot"""
    try:
        post("/rw/panel/ctrlstate?action=setctrlstate", {"ctrl-state": "motoron"})
        post("/rw/rapid/execution?action=resetpp")
        post(
            "/rw/rapid/execution?action=start",
            {
                "regain": "continue",
                "execmode": "continue",
                "cycle": "once",
                "condition": "none",
                "stopatbp": "disabled",
                "alltaskbytsp": "false"
            }
        )
        print(f"✓ RAPID started ({ROBOT_ARM} arm - {TASK_NAME})")
        return True
    except Exception as e:
        print(f"❌ Error starting RAPID: {e}")
        return False


def hold_rapid():
    """Stop RAPID execution"""
    try:
        post("/rw/rapid/execution?action=stop")
        print("RAPID stopped")
    except Exception as e:
        print(f"Error stopping RAPID: {e}")


# -----------------------------
# Robot Control Functions
# -----------------------------
def get_cartesian_position(mechunit=None):
    """Get current Cartesian position of robot"""
    if mechunit is None:
        mechunit = MECHUNIT

    try:
        response = get(f"/rw/motionsystem/mechunits/{mechunit}/robtarget?json=1")
        data = response.json()
        state = data.get("_embedded", {}).get("_state", [{}])[0]

        return {
            "x": float(state.get("x", 0)),
            "y": float(state.get("y", 0)),
            "z": float(state.get("z", 0)),
            "q1": float(state.get("q1", 0)),
            "q2": float(state.get("q2", 0)),
            "q3": float(state.get("q3", 0)),
            "q4": float(state.get("q4", 0))
        }
    except Exception as e:
        print(f"Error reading position: {e}")
        return None


def set_cartesian_target(x, y, z, q1, q2, q3, q4, task=None, module=None):
    """Send Cartesian target to robot"""
    if task is None:
        task = TASK_NAME
    if module is None:
        module = MODULE_NAME

    robtarget_value = f"[[{x},{y},{z}],[{q1},{q2},{q3},{q4}],[0,0,0,4],[-101.964,9E9,9E9,9E9,9E9,9E9]]"

    try:
        endpoint = f"/rw/rapid/symbol/data/RAPID/{task}/{module}/target?action=set"
        post(endpoint, {"value": robtarget_value})
        return True
    except Exception as e:
        print(f"Error setting target: {e}")
        return False


def is_position_safe(x, y, z):
    """Check if position is within workspace limits"""
    if not (WORKSPACE_LIMITS['x_min'] <= x <= WORKSPACE_LIMITS['x_max']):
        return False, f"X={x:.1f} out of range [{WORKSPACE_LIMITS['x_min']}, {WORKSPACE_LIMITS['x_max']}]"
    if not (WORKSPACE_LIMITS['y_min'] <= y <= WORKSPACE_LIMITS['y_max']):
        return False, f"Y={y:.1f} out of range [{WORKSPACE_LIMITS['y_min']}, {WORKSPACE_LIMITS['y_max']}]"
    if not (WORKSPACE_LIMITS['z_min'] <= z <= WORKSPACE_LIMITS['z_max']):
        return False, f"Z={z:.1f} out of range [{WORKSPACE_LIMITS['z_min']}, {WORKSPACE_LIMITS['z_max']}]"
    return True, "Position is safe"


# -----------------------------
# High-Level Robot API for LLM
# -----------------------------
class RobotController:
    def __init__(self):
        self.current_position = None
        self.update_position()

    def update_position(self):
        """Update current position from robot"""
        self.current_position = get_cartesian_position()
        return self.current_position

    def get_status(self):
        """Get current robot status as a string"""
        pos = self.update_position()
        if not pos:
            return "Error: Cannot read robot position"

        return f"""Current Robot Status ({ROBOT_ARM} Arm):
Position: X={pos['x']:.1f}mm, Y={pos['y']:.1f}mm, Z={pos['z']:.1f}mm
Orientation: q1={pos['q1']:.4f}, q2={pos['q2']:.4f}, q3={pos['q3']:.4f}, q4={pos['q4']:.4f}

Workspace Limits (DO NOT EXCEED):
  X: {WORKSPACE_LIMITS['x_min']} to {WORKSPACE_LIMITS['x_max']} mm (backward to forward)
  Y: {WORKSPACE_LIMITS['y_min']} to {WORKSPACE_LIMITS['y_max']} mm (left to right)
  Z: {WORKSPACE_LIMITS['z_min']} to {WORKSPACE_LIMITS['z_max']} mm (down to up)"""

    def move_to(self, x, y, z, description=""):
        """Move to absolute position (keeps current orientation)"""
        safe, message = is_position_safe(x, y, z)
        if not safe:
            return False, f"REJECTED: {message}"

        current = self.current_position
        if not current:
            return False, "Error: Cannot read current orientation"

        print(f"  → Moving to X={x:.1f}, Y={y:.1f}, Z={z:.1f} ({description})")
        success = set_cartesian_target(
            x, y, z,
            current['q1'], current['q2'], current['q3'], current['q4']
        )

        if success:
            time.sleep(0.5)
            self.update_position()
            return True, f"Success: Moved to X={x:.1f}, Y={y:.1f}, Z={z:.1f}"
        else:
            return False, "Error: Failed to send command"

    def rotate_wrist(self, degrees):
        """Rotate gripper around Z axis (like turning a doorknob)"""
        current = self.current_position
        if not current:
            return False, "Error: Cannot read current position"

        q = (current['q1'], current['q2'], current['q3'], current['q4'])
        q_new = rotate_quaternion(q, (0, 0, 1), degrees)

        print(f"  ⟲ Rotating wrist {degrees}° around Z axis")
        success = set_cartesian_target(
            current['x'], current['y'], current['z'],
            q_new[0], q_new[1], q_new[2], q_new[3]
        )

        if success:
            time.sleep(0.5)
            self.update_position()
            return True, f"Success: Rotated wrist {degrees}°"
        else:
            return False, "Error: Failed to rotate wrist"

    def tilt_gripper(self, degrees):
        """Tilt gripper forward(+) or backward(-)"""
        current = self.current_position
        if not current:
            return False, "Error: Cannot read current position"

        q = (current['q1'], current['q2'], current['q3'], current['q4'])
        q_new = rotate_quaternion(q, (0, 1, 0), degrees)

        direction = "forward" if degrees > 0 else "backward"
        print(f"  ⤴ Tilting gripper {abs(degrees)}° {direction}")
        success = set_cartesian_target(
            current['x'], current['y'], current['z'],
            q_new[0], q_new[1], q_new[2], q_new[3]
        )

        if success:
            time.sleep(0.5)
            self.update_position()
            return True, f"Success: Tilted gripper {degrees}°"
        else:
            return False, "Error: Failed to tilt gripper"

    def roll_gripper(self, degrees):
        """Roll gripper left(-) or right(+)"""
        current = self.current_position
        if not current:
            return False, "Error: Cannot read current position"

        q = (current['q1'], current['q2'], current['q3'], current['q4'])
        q_new = rotate_quaternion(q, (1, 0, 0), degrees)

        direction = "right" if degrees > 0 else "left"
        print(f"  ⤸ Rolling gripper {abs(degrees)}° {direction}")
        success = set_cartesian_target(
            current['x'], current['y'], current['z'],
            q_new[0], q_new[1], q_new[2], q_new[3]
        )

        if success:
            time.sleep(0.5)
            self.update_position()
            return True, f"Success: Rolled gripper {degrees}°"
        else:
            return False, "Error: Failed to roll gripper"


# -----------------------------
# LLM Integration (Gemini)
# -----------------------------
class LLMRobotChat:
    def __init__(self):
        self.robot = RobotController()
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.conversation_history = []
        self.initialize_system_prompt()

    def initialize_system_prompt(self):
        """Initialize system prompt"""
        self.system_prompt = f"""You are controlling an ABB YuMi robot's {ROBOT_ARM.lower()} arm. You must calculate exact coordinates based on natural language.

COORDINATE SYSTEM & AXIS MAPPINGS:
- X axis controls FORWARD/BACKWARD movement:
  * Negative X = Move BACKWARD (away from user)
  * Positive X = Move FORWARD (toward user)
  * Range: {WORKSPACE_LIMITS['x_min']} to {WORKSPACE_LIMITS['x_max']} mm

- Y axis controls LEFT/RIGHT movement:
  * Negative Y = Move LEFT
  * Positive Y = Move RIGHT
  * Range: {WORKSPACE_LIMITS['y_min']} to {WORKSPACE_LIMITS['y_max']} mm

- Z axis controls UP/DOWN movement:
  * Lower Z = Move DOWN
  * Higher Z = Move UP
  * Range: {WORKSPACE_LIMITS['z_min']} to {WORKSPACE_LIMITS['z_max']} mm

DISTANCE INTERPRETATION:
- "a bit" / "slightly" = 20-30mm
- "some" / "moderately" = 50mm
- "significantly" / "a lot" = 80-100mm
- No modifier (e.g., "move up") = 50mm default

AVAILABLE COMMAND:
robot.move_to(x, y, z, "description")

HOW TO CALCULATE POSITIONS:

1. READ CURRENT POSITION from the status (will be provided)
2. DETERMINE which axis/axes to adjust:
   - "up" or "down" → adjust Z only
   - "left" or "right" → adjust Y only
   - "forward" or "backward" → adjust X only
   - "up and left" → adjust both Z and Y
   - "forward and down" → adjust both X and Z

3. CALCULATE new coordinate:
   - Keep unchanged axes at current value
   - Adjust specified axes by appropriate amount
   - ENSURE result stays within workspace limits

4. OUTPUT the robot.move_to() command with calculated values

EXAMPLES:

Current: X=0.0, Y=-300.0, Z=150.0

User: "Move up a bit"
Think: "up" = increase Z by ~30mm, keep X and Y same
Response: robot.move_to(0.0, -300.0, 180.0, "moving up a bit")

User: "Move to the left"
Think: "left" = decrease Y by ~50mm, keep X and Z same
Response: robot.move_to(0.0, -350.0, 150.0, "moving left")

User: "Move to the right"
Think: "right" = increase Y by ~50mm, keep X and Z same
Response: robot.move_to(0.0, -250.0, 150.0, "moving right")

User: "Go forward"
Think: "forward" = increase X by ~50mm, keep Y and Z same
Response: robot.move_to(50.0, -300.0, 150.0, "moving forward")

User: "Go backward"
Think: "backward" = decrease X by ~50mm, keep Y and Z same
Response: robot.move_to(-50.0, -300.0, 150.0, "moving backward")

User: "Go forward and down"
Think: "forward" = increase X by ~50mm, "down" = decrease Z by ~50mm, keep Y same
Response: robot.move_to(50.0, -300.0, 100.0, "moving forward and down")

User: "Look up to your left"
Think: "up" = increase Z significantly, "left" = decrease Y, keep X same
Response: robot.move_to(0.0, -380.0, 230.0, "looking up-left")

User: "Point straight ahead at medium height"
Think: "straight ahead" = max X safely, "medium height" = middle of Z range
Response: robot.move_to(120.0, -300.0, 185.0, "pointing forward at medium height")

User: "Move to top-right corner"
Think: "top" = max Z, "right" = max Y, forward X position
Response: robot.move_to(80.0, -140.0, 270.0, "top-right corner")

ORIENTATION COMMANDS (optional):
- robot.rotate_wrist(degrees) - positive = counterclockwise
- robot.tilt_gripper(degrees) - positive = forward tilt
- robot.roll_gripper(degrees) - positive = roll right

CRITICAL RULES:
1. ALWAYS output robot.move_to(x, y, z, "description") with calculated numbers
2. NEVER exceed workspace limits
3. Use current position as starting point for calculations
4. Output ONLY the Python command, no explanations
5. For combined movements, adjust multiple axes in ONE move_to command"""

        print(f"✓ Gemini LLM initialized for {ROBOT_ARM} arm with dynamic coordinate calculation!")

    def send_message(self, user_message):
        """Send message to LLM and get response"""
        print(f"\n{'=' * 60}")
        print(f"USER: {user_message}")
        print(f"{'=' * 60}")

        status = self.robot.get_status()
        full_prompt = f"""{self.system_prompt}

{status}

User request: {user_message}

Calculate the coordinates and respond with ONLY the Python command:"""

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=full_prompt
            )

            llm_response = response.text.strip()

            # Remove markdown code blocks if present
            llm_response = llm_response.replace('```python', '').replace('```', '').strip()

            print(f"\nGEMINI: {llm_response}")

            # Execute commands
            if "robot." in llm_response:
                print(f"\n{'=' * 60}")
                print("EXECUTING...")
                print(f"{'=' * 60}")
                self.execute_commands(llm_response)
            else:
                print("\n⚠️ No robot commands found in response")

            return llm_response

        except Exception as e:
            print(f"❌ Error communicating with Gemini: {e}")
            return None

    def execute_commands(self, llm_response):
        """Extract and execute robot commands"""
        lines = llm_response.split('\n')

        for line in lines:
            line = line.strip()

            if line.startswith("robot."):
                command = line
                print(f"\n📤 {command}")

                try:
                    result = eval(command, {"robot": self.robot})

                    if isinstance(result, tuple):
                        success, message = result
                        if success:
                            print(f"✅ {message}")
                        else:
                            print(f"❌ {message}")
                    else:
                        print(f"✅ Done")

                except Exception as e:
                    print(f"❌ Error executing: {e}")

    def start_interactive_session(self):
        """Start interactive chat"""
        print("\n" + "=" * 60)
        print(f"🤖 GEMINI ROBOT CONTROL - {ROBOT_ARM} Arm")
        print("=" * 60)
        print("\nTry commands like:")
        print("  - 'Move up a bit'")
        print("  - 'Go to the left'")
        print("  - 'Move forward'")
        print("  - 'Point forward and down'")
        print("  - 'Look up to your left'")
        print("  - 'Move to the top-right corner'")
        print("  - 'Reach forward at medium height'")
        print("\nType 'quit' to exit, 'status' for robot status.\n")

        while True:
            try:
                user_input = input("\nYOU: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    print("Exiting...")
                    break

                if user_input.lower() == 'status':
                    print(self.robot.get_status())
                    continue

                self.send_message(user_input)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


# -----------------------------
# Shutdown handling
# -----------------------------
def shutdown_handler(sig, frame):
    print("\nStopping robot...")
    hold_rapid()
    SESSION.close()
    sys.exit(0)


# -----------------------------
# Main
# -----------------------------
def main():
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print(f"\n{'=' * 60}")
    print(f"ABB YuMi Robot Controller - {ROBOT_ARM} Arm")
    print(f"Task: {TASK_NAME}, Module: {MODULE_NAME}, MechUnit: {MECHUNIT}")
    print(f"{'=' * 60}")

    # START RAPID
    print("\nStarting RAPID on robot...")
    if not start_rapid():
        print("❌ Failed to start RAPID. Check robot connection and controller state.")
        return

    time.sleep(1)

    # Initialize LLM chat
    print("\nInitializing Gemini...")
    llm_chat = LLMRobotChat()

    # Start interactive session
    llm_chat.start_interactive_session()


if __name__ == "__main__":
    main()
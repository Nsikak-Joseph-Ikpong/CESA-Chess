import requests
import time
import signal
import sys
from requests.auth import HTTPDigestAuth
import json
import math
import cv2
import numpy as np
import threading

# -----------------------------
# Robot connection details
# -----------------------------
# ROBOT_IP = "127.0.0.1"
ROBOT_IP = "192.168.125.1"
USERNAME = "Default User"
PASSWORD = "robotics"

BASE_URL = f"http://{ROBOT_IP}"
AUTH = HTTPDigestAuth(USERNAME, PASSWORD)

HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded"
}

# Create a persistent session to avoid "too many sessions" error
SESSION = requests.Session()
SESSION.auth = AUTH


# -----------------------------
# Helper HTTP functions
# -----------------------------
def post(endpoint, data=None):
    r = SESSION.post(
        BASE_URL + endpoint,
        headers=HEADERS,
        data=data
    )
    if not r.ok:
        print(f"Error response: {r.status_code}")
        print(f"Response text: {r.text}")
    r.raise_for_status()
    return r


def get(endpoint):
    r = SESSION.get(
        BASE_URL + endpoint
    )
    if not r.ok:
        print(f"Error response: {r.status_code}")
        print(f"Response text: {r.text}")
    r.raise_for_status()
    return r


# -----------------------------
# RAPID control
# -----------------------------
def start_rapid():
    post(
        "/rw/panel/ctrlstate?action=setctrlstate",
        {"ctrl-state": "motoron"}
    )
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
    print("RAPID started (looping)")


def hold_rapid():
    post("/rw/rapid/execution?action=hold")
    print("RAPID held")


# -----------------------------
# Speed control
# -----------------------------
def set_speed(percent):
    if not (0 <= percent <= 100):
        print("Speed must be between 0 and 100")
        return
    post(
        "/rw/panel/speedratio?action=setspeedratio",
        {"speed-ratio": percent}
    )
    print(f"Speed set to {percent}%")


# -----------------------------
# Position reading
# -----------------------------
def list_mechunits():
    try:
        response = get("/rw/motionsystem/mechunits?json=1")
        data = response.json()
        mechunits = data.get("_embedded", {}).get("_state", [])

        print("\nAvailable mechanical units:")
        unit_names = []
        for unit in mechunits:
            name = unit.get("_title", "Unknown")
            mode = unit.get("mode", "Unknown")
            print(f"  - {name} (Mode: {mode})")
            unit_names.append(name)

        return unit_names
    except Exception as e:
        print(f"Error listing mechanical units: {e}")
        return []


def get_joint_positions(mechunit=None):
    if mechunit is None:
        mechunit = "ROB_R"

    try:
        response = get(f"/rw/motionsystem/mechunits/{mechunit}/jointtarget?json=1")
        data = response.json()
        state = data.get("_embedded", {}).get("_state", [{}])[0]

        joints = []
        for i in range(1, 8):
            key = f"rax_{i}"
            if key in state:
                value = float(state[key])
                joints.append(value)

        print(f"\nJoint positions for {mechunit}:")
        for i, pos in enumerate(joints, 1):
            print(f"  Joint {i}: {pos:.3f}°")

        ext_axes = []
        for letter in ['a', 'b', 'c', 'd', 'e', 'f']:
            key = f"eax_{letter}"
            if key in state:
                value = float(state[key])
                if abs(value) < 1e6:
                    ext_axes.append((letter, value))

        if ext_axes:
            print(f"\nExternal axes for {mechunit}:")
            for letter, pos in ext_axes:
                print(f"  Ext {letter.upper()}: {pos:.3f}°")

        return joints
    except Exception as e:
        print(f"Error parsing joint positions: {e}")
        return None


def get_cartesian_position(mechunit=None, verbose=True):
    if mechunit is None:
        mechunit = "ROB_R"

    try:
        response = get(f"/rw/motionsystem/mechunits/{mechunit}/robtarget?json=1")
        data = response.json()
        state = data.get("_embedded", {}).get("_state", [{}])[0]

        x = float(state.get("x", 0))
        y = float(state.get("y", 0))
        z = float(state.get("z", 0))
        q1 = float(state.get("q1", 0))
        q2 = float(state.get("q2", 0))
        q3 = float(state.get("q3", 0))
        q4 = float(state.get("q4", 0))

        if verbose:
            print(f"\nCartesian position for {mechunit}:")
            print(f"  Position (mm):")
            print(f"    X: {x:.2f}")
            print(f"    Y: {y:.2f}")
            print(f"    Z: {z:.2f}")
            print(f"  Orientation (quaternion):")
            print(f"    q1: {q1:.4f}")
            print(f"    q2: {q2:.4f}")
            print(f"    q3: {q3:.4f}")
            print(f"    q4: {q4:.4f}")

        return {
            "x": x,
            "y": y,
            "z": z,
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "q4": q4
        }
    except Exception as e:
        if verbose:
            print(f"Error parsing Cartesian position: {e}")
        return None


def get_all_positions():
    print("\n" + "=" * 40)
    print("RIGHT ARM (ROB_R)")
    print("=" * 40)
    get_joint_positions("ROB_R")

    print("\n" + "=" * 40)
    print("LEFT ARM (ROB_L)")
    print("=" * 40)
    get_joint_positions("ROB_L")


def get_all_cartesian():
    print("\n" + "=" * 40)
    print("RIGHT ARM (ROB_R)")
    print("=" * 40)
    get_cartesian_position("ROB_R")

    print("\n" + "=" * 40)
    print("LEFT ARM (ROB_L)")
    print("=" * 40)
    get_cartesian_position("ROB_L")


# -----------------------------
# Position writing - RIGHT ARM
# -----------------------------
def set_cartesian_target(x, y, z, q1=None, q2=None, q3=None, q4=None, task="T_ROB_R", module="Module1",
                         variable="target", verbose=True):
    if q1 is None or q2 is None or q3 is None or q4 is None:
        if verbose:
            print("No orientation provided, using current orientation...")
        mechunit = "ROB_R" if task == "T_ROB_R" else "ROB_L"
        current = get_cartesian_position(mechunit, verbose=False)
        if current:
            q1 = current["q1"]
            q2 = current["q2"]
            q3 = current["q3"]
            q4 = current["q4"]
        else:
            if verbose:
                print("Could not get current orientation, using default")
            q1, q2, q3, q4 = 1, 0, 0, 0

    robtarget_value = f"[[{x},{y},{z}],[{q1},{q2},{q3},{q4}],[0,0,0,4],[-101.964,9E9,9E9,9E9,9E9,9E9]]"

    if verbose:
        print(f"\nSetting target for {task}:")
        print(f"  Position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
        print(f"  Orientation: q1={q1:.4f}, q2={q2:.4f}, q3={q3:.4f}, q4={q4:.4f}")

    try:
        # Set the target position
        endpoint = f"/rw/rapid/symbol/data/RAPID/{task}/{module}/{variable}?action=set&mastership=implicit"
        post(endpoint, {"value": robtarget_value})

        # Set new_target flag to TRUE to trigger the move
        flag_endpoint = f"/rw/rapid/symbol/data/RAPID/{task}/{module}/new_target?action=set&mastership=implicit"
        post(flag_endpoint, {"value": "TRUE"})

        if verbose:
            print(f"✓ Target written and move triggered")
        return True
    except Exception as e:
        if verbose:
            print(f"✗ Error writing target: {e}")
        return False


# -----------------------------
# Position writing - LEFT ARM (UPDATED WITH FLAG)
# -----------------------------
def set_cartesian_target_left(x, y, z, q1=None, q2=None, q3=None, q4=None,
                              task="T_ROB_L", module="Module1",
                              variable="target", verbose=True):
    """Set target for left arm"""
    if q1 is None or q2 is None or q3 is None or q4 is None:
        current = get_cartesian_position("ROB_L", verbose=False)
        if current:
            q1 = current["q1"]
            q2 = current["q2"]
            q3 = current["q3"]
            q4 = current["q4"]
        else:
            q1, q2, q3, q4 = 1, 0, 0, 0

    robtarget_value = f"[[{x},{y},{z}],[{q1},{q2},{q3},{q4}],[0,0,0,4],[118.745,9E9,9E9,9E9,9E9,9E9]]"

    try:
        # Just set the target - RAPID will detect the position change
        endpoint = f"/rw/rapid/symbol/data/RAPID/{task}/{module}/{variable}?action=set&mastership=implicit"
        post(endpoint, {"value": robtarget_value})

        if verbose:
            print(f"✓ Target written")
        return True
    except Exception as e:
        if verbose:
            print(f"✗ Error: {e}")
        return False


def check_move_complete(task="T_ROB_L", module="Module1"):
    """Check if robot has completed the move"""
    try:
        response = get(f"/rw/rapid/symbol/data/RAPID/{task}/{module}/move_busy?json=1")
        data = response.json()
        state = data.get("_embedded", {}).get("_state", [{}])[0]
        value = state.get("value", "FALSE")
        # Return True if NOT busy (i.e., move is complete)
        return "FALSE" in value.upper()
    except Exception as e:
        print(f"Error checking move status: {e}")
        return True  # Assume complete on error


def move_to(x, y, z, arm="right", verbose=True):
    task = "T_ROB_R" if arm == "right" else "T_ROB_L"
    module = "Module1"
    if arm == "right":
        return set_cartesian_target(x, y, z, task=task, module=module, verbose=verbose)
    else:
        return set_cartesian_target_left(x, y, z, task=task, module=module, verbose=verbose)


# -----------------------------
# Quaternion math for orientation control
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
# Keyboard jog control (step-by-step)
# -----------------------------
def keyboard_jog(arm="right"):
    """
    Keyboard-controlled robot jogging for workspace exploration
    WITH ORIENTATION CONTROL
    """
    arm_name = "RIGHT ARM" if arm == "right" else "LEFT ARM"
    mechunit = "ROB_R" if arm == "right" else "ROB_L"

    print("\n" + "=" * 60)
    print(f"KEYBOARD JOG MODE WITH ORIENTATION CONTROL - {arm_name}")
    print("=" * 60)
    print("\nControls:")
    print("  Position:")
    print("    w      : Move forward (Y+)")
    print("    s      : Move back (Y-)")
    print("    a      : Move left (X-)")
    print("    d      : Move right (X+)")
    print("    zu     : Move up (Z+)")
    print("    zd     : Move down (Z-)")
    print("  Orientation (rotation):")
    print("    rx+    : Rotate +around X axis (roll)")
    print("    rx-    : Rotate -around X axis")
    print("    ry+    : Rotate +around Y axis (pitch)")
    print("    ry-    : Rotate -around Y axis")
    print("    rz+    : Rotate +around Z axis (yaw)")
    print("    rz-    : Rotate -around Z axis")
    print("  Step size:")
    print("    +      : Increase step size")
    print("    -      : Decrease step size")
    print("    r+     : Increase rotation step")
    print("    r-     : Decrease rotation step")
    print("  Recording:")
    print("    rec    : Record current position + orientation")
    print("    c      : Show current position + orientation")
    print("    list   : Show all recorded positions")
    print("  Exit:")
    print("    done   : Calculate workspace and save")
    print("    q      : Quit without saving")
    print("\n")

    # Get starting position and orientation
    current = get_cartesian_position(mechunit, verbose=False)
    if not current:
        print("❌ Could not read current position!")
        return

    x, y, z = current['x'], current['y'], current['z']
    q1, q2, q3, q4 = current['q1'], current['q2'], current['q3'], current['q4']

    step_size = 10  # mm for position
    rot_step = 5  # degrees for rotation
    recorded = {}

    print(f"Starting position: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
    print(f"Starting orientation: q1={q1:.3f}, q2={q2:.3f}, q3={q3:.3f}, q4={q4:.3f}")
    print(f"Position step: {step_size}mm, Rotation step: {rot_step}°\n")
    print("Ready! Use w/a/s/d/zu/zd to move, rx+/ry+/rz+ to rotate.\n")

    while True:
        cmd = input(f"[X={x:.1f} Y={y:.1f} Z={z:.1f}] > ").strip().lower()

        if cmd == "q":
            print("Quitting without saving...")
            return None

        elif cmd == "done":
            # Done - calculate workspace
            break

        elif cmd == "c":
            # Show current position
            actual = get_cartesian_position(mechunit, verbose=True)
            if actual:
                x, y, z = actual['x'], actual['y'], actual['z']
                q1, q2, q3, q4 = actual['q1'], actual['q2'], actual['q3'], actual['q4']
            continue

        elif cmd == "list":
            # Show recorded positions
            print("\nRecorded positions:")
            if not recorded:
                print("  (none)")
            for name, pos in recorded.items():
                print(
                    f"  {name}: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f}, q=[{pos[3]:.2f},{pos[4]:.2f},{pos[5]:.2f},{pos[6]:.2f}]")
            print()
            continue

        elif cmd == "rec":
            # Record position + orientation
            name = input("Position name (e.g., max_x, min_y, rotated): ").strip()
            if name:
                recorded[name] = (x, y, z, q1, q2, q3, q4)
                print(f"✓ Recorded '{name}': X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
                print(f"  Orientation: q1={q1:.3f}, q2={q2:.3f}, q3={q3:.3f}, q4={q4:.3f}")
            continue

        elif cmd == "+":
            step_size = min(step_size + 5, 50)
            print(f"Position step: {step_size}mm")
            continue

        elif cmd == "-":
            step_size = max(step_size - 5, 1)
            print(f"Position step: {step_size}mm")
            continue

        elif cmd == "r+":
            rot_step = min(rot_step + 5, 45)
            print(f"Rotation step: {rot_step}°")
            continue

        elif cmd == "r-":
            rot_step = max(rot_step - 5, 1)
            print(f"Rotation step: {rot_step}°")
            continue

        # Position movement commands
        elif cmd in ["up", "u", "w"]:
            y_new = y + step_size
            print(f"Moving Y: {y:.1f} → {y_new:.1f} (forward)")
            if arm == "right":
                success = set_cartesian_target(x, y_new, z, q1, q2, q3, q4, verbose=False)
            else:
                success = set_cartesian_target_left(x, y_new, z, q1, q2, q3, q4, verbose=False)

            if success:
                time.sleep(0.5)
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    y = actual['y']
                    print(f"✓ Moved to Y={y:.1f}")
            else:
                print("✗ Move failed")

        elif cmd in ["down", "s"]:
            y_new = y - step_size
            print(f"Moving Y: {y:.1f} → {y_new:.1f} (back)")
            if arm == "right":
                success = set_cartesian_target(x, y_new, z, q1, q2, q3, q4, verbose=False)
            else:
                success = set_cartesian_target_left(x, y_new, z, q1, q2, q3, q4, verbose=False)

            if success:
                time.sleep(0.5)
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    y = actual['y']
                    print(f"✓ Moved to Y={y:.1f}")
            else:
                print("✗ Move failed")

        elif cmd in ["left", "l", "a"]:
            x_new = x - step_size
            print(f"Moving X: {x:.1f} → {x_new:.1f} (left)")
            if arm == "right":
                success = set_cartesian_target(x_new, y, z, q1, q2, q3, q4, verbose=False)
            else:
                success = set_cartesian_target_left(x_new, y, z, q1, q2, q3, q4, verbose=False)

            if success:
                time.sleep(0.5)
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    x = actual['x']
                    print(f"✓ Moved to X={x:.1f}")
            else:
                print("✗ Move failed")

        elif cmd in ["right", "d"]:
            x_new = x + step_size
            print(f"Moving X: {x:.1f} → {x_new:.1f} (right)")
            if arm == "right":
                success = set_cartesian_target(x_new, y, z, q1, q2, q3, q4, verbose=False)
            else:
                success = set_cartesian_target_left(x_new, y, z, q1, q2, q3, q4, verbose=False)

            if success:
                time.sleep(0.5)
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    x = actual['x']
                    print(f"✓ Moved to X={x:.1f}")
            else:
                print("✗ Move failed")

        elif cmd in ["zu", "pageup", "pgup"]:
            z_new = z + step_size
            print(f"Moving Z: {z:.1f} → {z_new:.1f} (up)")
            if arm == "right":
                success = set_cartesian_target(x, y, z_new, q1, q2, q3, q4, verbose=False)
            else:
                success = set_cartesian_target_left(x, y, z_new, q1, q2, q3, q4, verbose=False)

            if success:
                time.sleep(0.5)
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    z = actual['z']
                    print(f"✓ Moved to Z={z:.1f}")
            else:
                print("✗ Move failed")

        elif cmd in ["zd", "pagedown", "pgdn"]:
            z_new = z - step_size
            print(f"Moving Z: {z:.1f} → {z_new:.1f} (down)")
            if arm == "right":
                success = set_cartesian_target(x, y, z_new, q1, q2, q3, q4, verbose=False)
            else:
                success = set_cartesian_target_left(x, y, z_new, q1, q2, q3, q4, verbose=False)

            if success:
                time.sleep(0.5)
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    z = actual['z']
                    print(f"✓ Moved to Z={z:.1f}")
            else:
                print("✗ Move failed")

        # Orientation rotation commands
        elif cmd == "rx+":
            print(f"Rotating +{rot_step}° around X axis (roll)")
            q_new = rotate_quaternion((q1, q2, q3, q4), (1, 0, 0), rot_step)
            q1_n, q2_n, q3_n, q4_n = q_new
            if arm == "right":
                success = set_cartesian_target(x, y, z, q1_n, q2_n, q3_n, q4_n, verbose=False)
            else:
                success = set_cartesian_target_left(x, y, z, q1_n, q2_n, q3_n, q4_n, verbose=False)

            if success:
                time.sleep(0.5)
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    q1, q2, q3, q4 = actual['q1'], actual['q2'], actual['q3'], actual['q4']
                    print(f"✓ Rotated")
            else:
                print("✗ Rotation failed")

        elif cmd == "rx-":
            print(f"Rotating -{rot_step}° around X axis (roll)")
            q_new = rotate_quaternion((q1, q2, q3, q4), (1, 0, 0), -rot_step)
            q1_n, q2_n, q3_n, q4_n = q_new
            if arm == "right":
                success = set_cartesian_target(x, y, z, q1_n, q2_n, q3_n, q4_n, verbose=False)
            else:
                success = set_cartesian_target_left(x, y, z, q1_n, q2_n, q3_n, q4_n, verbose=False)

            if success:
                time.sleep(0.5)
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    q1, q2, q3, q4 = actual['q1'], actual['q2'], actual['q3'], actual['q4']
                    print(f"✓ Rotated")
            else:
                print("✗ Rotation failed")

        elif cmd == "ry+":
            print(f"Rotating +{rot_step}° around Y axis (pitch)")
            q_new = rotate_quaternion((q1, q2, q3, q4), (0, 1, 0), rot_step)
            q1_n, q2_n, q3_n, q4_n = q_new
            if arm == "right":
                success = set_cartesian_target(x, y, z, q1_n, q2_n, q3_n, q4_n, verbose=False)
            else:
                success = set_cartesian_target_left(x, y, z, q1_n, q2_n, q3_n, q4_n, verbose=False)

            if success:
                time.sleep(0.5)
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    q1, q2, q3, q4 = actual['q1'], actual['q2'], actual['q3'], actual['q4']
                    print(f"✓ Rotated")
            else:
                print("✗ Rotation failed")

        elif cmd == "ry-":
            print(f"Rotating -{rot_step}° around Y axis (pitch)")
            q_new = rotate_quaternion((q1, q2, q3, q4), (0, 1, 0), -rot_step)
            q1_n, q2_n, q3_n, q4_n = q_new
            if arm == "right":
                success = set_cartesian_target(x, y, z, q1_n, q2_n, q3_n, q4_n, verbose=False)
            else:
                success = set_cartesian_target_left(x, y, z, q1_n, q2_n, q3_n, q4_n, verbose=False)

            if success:
                time.sleep(0.5)
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    q1, q2, q3, q4 = actual['q1'], actual['q2'], actual['q3'], actual['q4']
                    print(f"✓ Rotated")
            else:
                print("✗ Rotation failed")

        elif cmd == "rz+":
            print(f"Rotating +{rot_step}° around Z axis (yaw)")
            q_new = rotate_quaternion((q1, q2, q3, q4), (0, 0, 1), rot_step)
            q1_n, q2_n, q3_n, q4_n = q_new
            if arm == "right":
                success = set_cartesian_target(x, y, z, q1_n, q2_n, q3_n, q4_n, verbose=False)
            else:
                success = set_cartesian_target_left(x, y, z, q1_n, q2_n, q3_n, q4_n, verbose=False)

            if success:
                time.sleep(0.5)
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    q1, q2, q3, q4 = actual['q1'], actual['q2'], actual['q3'], actual['q4']
                    print(f"✓ Rotated")
            else:
                print("✗ Rotation failed")

        elif cmd == "rz-":
            print(f"Rotating -{rot_step}° around Z axis (yaw)")
            q_new = rotate_quaternion((q1, q2, q3, q4), (0, 0, 1), -rot_step)
            q1_n, q2_n, q3_n, q4_n = q_new
            if arm == "right":
                success = set_cartesian_target(x, y, z, q1_n, q2_n, q3_n, q4_n, verbose=False)
            else:
                success = set_cartesian_target_left(x, y, z, q1_n, q2_n, q3_n, q4_n, verbose=False)

            if success:
                time.sleep(0.5)
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    q1, q2, q3, q4 = actual['q1'], actual['q2'], actual['q3'], actual['q4']
                    print(f"✓ Rotated")
            else:
                print("✗ Rotation failed")

        else:
            print("Unknown command. Available: w/s/a/d/zu/zd (position), rx+/rx-/ry+/ry-/rz+/rz- (rotation)")

    # Calculate workspace from recorded positions
    if len(recorded) >= 6:
        print("\n" + "=" * 60)
        print("CALCULATING WORKSPACE LIMITS")
        print("=" * 60)

        all_x = [p[0] for p in recorded.values()]
        all_y = [p[1] for p in recorded.values()]
        all_z = [p[2] for p in recorded.values()]

        margin = 30

        limits = {
            'x_min': min(all_x) + margin,
            'x_max': max(all_x) - margin,
            'y_min': min(all_y) + margin,
            'y_max': max(all_y) - margin,
            'z_min': min(all_z) + margin,
            'z_max': max(all_z) - margin
        }

        print(f"\nWorkspace limits (with {margin}mm safety margin):")
        print(f"  X: {limits['x_min']:.0f} to {limits['x_max']:.0f} mm")
        print(f"  Y: {limits['y_min']:.0f} to {limits['y_max']:.0f} mm")
        print(f"  Z: {limits['z_min']:.0f} to {limits['z_max']:.0f} mm")

        # Save to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"workspace_keyboard_{arm}_{timestamp}.txt"

        with open(filename, 'w') as f:
            f.write(f"KEYBOARD JOG WORKSPACE DEFINITION - {arm_name} (WITH ORIENTATION)\n")
            f.write("=" * 60 + "\n\n")
            f.write("Recorded positions:\n")
            for name, pos in recorded.items():
                f.write(f"  {name}: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f}\n")
                f.write(f"         q1={pos[3]:.3f}, q2={pos[4]:.3f}, q3={pos[5]:.3f}, q4={pos[6]:.3f}\n")
            f.write("\n")
            f.write(f"Workspace limits (with {margin}mm margin):\n")
            f.write(f"X: {limits['x_min']:.0f} to {limits['x_max']:.0f} mm\n")
            f.write(f"Y: {limits['y_min']:.0f} to {limits['y_max']:.0f} mm\n")
            f.write(f"Z: {limits['z_min']:.0f} to {limits['z_max']:.0f} mm\n\n")

            f.write("Python validation code:\n")
            f.write("```python\n")
            f.write(f"def is_position_safe(x, y, z, arm='{arm}'):\n")
            f.write(f"    if arm == '{arm}':\n")
            f.write(f"        x_min, x_max = {limits['x_min']:.0f}, {limits['x_max']:.0f}\n")
            f.write(f"        y_min, y_max = {limits['y_min']:.0f}, {limits['y_max']:.0f}\n")
            f.write(f"        z_min, z_max = {limits['z_min']:.0f}, {limits['z_max']:.0f}\n")
            f.write("        if not (x_min <= x <= x_max): return False\n")
            f.write("        if not (y_min <= y <= y_max): return False\n")
            f.write("        if not (z_min <= z <= z_max): return False\n")
            f.write("        return True\n")
            f.write("```\n")

        print(f"\n✓ Saved to: {filename}")
        print("\nReady-to-use validation code saved in file!")
        return limits
    else:
        print(f"\n⚠️ Need at least 6 positions (max/min for X,Y,Z), only have {len(recorded)}")
        return None


# -----------------------------
# Camera Functions
# -----------------------------
def capture_gripper_camera(camera_url=None):
    """
    Capture image from right arm smart gripper camera
    Returns: numpy array (BGR format for OpenCV)
    """
    if camera_url is None:
        # Default endpoint - update this based on your robot
        camera_url = "/rw/camera/snapshot?camera=1"

    try:
        response = get(camera_url)

        if response.status_code == 200:
            # Convert to numpy array
            image_data = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            return image
        else:
            return None

    except Exception as e:
        return None


# -----------------------------
# Streaming keyboard jog control WITH CAMERA
# -----------------------------
def streaming_jog(arm="right"):
    """
    Streaming keyboard control with live camera feed
    """
    try:
        import keyboard
    except ImportError:
        print("\n❌ 'keyboard' library not installed!")
        print("Install with: pip install keyboard")
        print("Then run this script as administrator/root\n")
        return

    arm_name = "RIGHT ARM" if arm == "right" else "LEFT ARM"
    mechunit = "ROB_R" if arm == "right" else "ROB_L"

    print("\n" + "=" * 60)
    print(f"STREAMING KEYBOARD JOG MODE WITH CAMERA - {arm_name}")
    print("=" * 60)
    print("\n⚠️  This mode requires administrator/root privileges!")
    print("⚠️  Press ESC to exit streaming mode\n")
    print("\nControls (HOLD keys to move continuously):")
    print("  Position:")
    print("    W      : Move forward (Y+)")
    print("    S      : Move back (Y-)")
    print("    A      : Move left (X-)")
    print("    D      : Move right (X+)")
    print("    Q      : Move up (Z+)")
    print("    E      : Move down (Z-)")
    print("  Orientation:")
    print("    I/K    : Rotate around X axis (roll)")
    print("    J/L    : Rotate around Y axis (pitch)")
    print("    U/O    : Rotate around Z axis (yaw)")
    print("  Camera:")
    print("    C      : Toggle camera on/off")
    print("    V      : Save snapshot")
    print("  Info:")
    print("    G      : Print current position")
    print("    R      : Read actual position from robot")
    print("  Speed:")
    print("    +/-    : Increase/decrease speed")
    print("  Exit:")
    print("    ESC    : Exit streaming mode")
    print("\n")

    # Get starting position
    current = get_cartesian_position(mechunit, verbose=False)
    if not current:
        print("❌ Could not read current position!")
        return

    x, y, z = current['x'], current['y'], current['z']
    q1, q2, q3, q4 = current['q1'], current['q2'], current['q3'], current['q4']

    pos_speed = 5.0  # mm per update
    rot_speed = 2.0  # degrees per update
    update_rate = 0.1  # seconds (10 Hz)

    print(f"Starting position: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
    print(f"Speed: {pos_speed}mm/update, Rotation: {rot_speed}°/update")
    print(f"Update rate: {1 / update_rate:.0f} Hz\n")

    # Camera setup
    camera_enabled = True
    snapshot_count = 0
    last_camera_update = 0
    camera_update_rate = 0.2  # Update camera every 200ms (5 fps)

    # Test camera
    print("Testing camera connection...")
    test_image = capture_gripper_camera()
    if test_image is not None:
        print("✅ Camera connected!")
        cv2.namedWindow(f'YuMi {arm_name} Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'YuMi {arm_name} Camera', 640, 480)
    else:
        print("⚠️ Camera not found - continuing without camera")
        camera_enabled = False

    print("\nReady! Hold keys to move. Press ESC to exit.\n")

    last_g_press = 0
    last_r_press = 0
    last_c_press = 0
    last_v_press = 0

    try:
        while True:
            moved = False
            current_time = time.time()

            # Check for exit
            if keyboard.is_pressed('esc'):
                print("\nExiting streaming mode...")
                break

            # Toggle camera (debounced)
            if keyboard.is_pressed('c') and (current_time - last_c_press) > 0.5:
                camera_enabled = not camera_enabled
                status = "ON" if camera_enabled else "OFF"
                print(f"\n📷 Camera: {status}")
                if not camera_enabled:
                    cv2.destroyAllWindows()
                last_c_press = current_time

            # Save snapshot (debounced)
            if keyboard.is_pressed('v') and (current_time - last_v_press) > 0.5:
                if camera_enabled:
                    snapshot_image = capture_gripper_camera()
                    if snapshot_image is not None:
                        snapshot_count += 1
                        filename = f"gripper_snapshot_{arm}_{snapshot_count}.jpg"
                        cv2.imwrite(filename, snapshot_image)
                        print(f"\n📸 Snapshot saved: {filename}")
                last_v_press = current_time

            # Update camera feed
            if camera_enabled and (current_time - last_camera_update) > camera_update_rate:
                camera_image = capture_gripper_camera()
                if camera_image is not None:
                    # Add position overlay on image
                    overlay = camera_image.copy()

                    # Add text overlay with position info
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(overlay, f"X: {x:.1f}mm", (10, 30), font, 0.7, (0, 255, 0), 2)
                    cv2.putText(overlay, f"Y: {y:.1f}mm", (10, 60), font, 0.7, (0, 255, 0), 2)
                    cv2.putText(overlay, f"Z: {z:.1f}mm", (10, 90), font, 0.7, (0, 255, 0), 2)
                    cv2.putText(overlay, f"Speed: {pos_speed:.1f}mm/s", (10, 120), font, 0.6, (255, 255, 0), 2)

                    # Add crosshair in center
                    h, w = overlay.shape[:2]
                    cv2.line(overlay, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 255), 2)
                    cv2.line(overlay, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 255), 2)

                    cv2.imshow(f'YuMi {arm_name} Camera', overlay)
                    cv2.waitKey(1)

                last_camera_update = current_time

            # Print current position (debounced)
            if keyboard.is_pressed('g') and (current_time - last_g_press) > 0.5:
                print(f"\n📍 Current Position:")
                print(f"   X: {x:.2f} mm")
                print(f"   Y: {y:.2f} mm")
                print(f"   Z: {z:.2f} mm")
                print(f"   q1: {q1:.4f}")
                print(f"   q2: {q2:.4f}")
                print(f"   q3: {q3:.4f}")
                print(f"   q4: {q4:.4f}")
                last_g_press = current_time
                time.sleep(0.1)

            # Read actual position from robot (debounced)
            if keyboard.is_pressed('r') and (current_time - last_r_press) > 0.5:
                print(f"\n🔄 Reading actual position from {arm_name}...")
                actual = get_cartesian_position(mechunit, verbose=False)
                if actual:
                    x, y, z = actual['x'], actual['y'], actual['z']
                    q1, q2, q3, q4 = actual['q1'], actual['q2'], actual['q3'], actual['q4']
                    print(f"   X: {x:.2f} mm")
                    print(f"   Y: {y:.2f} mm")
                    print(f"   Z: {z:.2f} mm")
                    print(f"   q1: {q1:.4f}")
                    print(f"   q2: {q2:.4f}")
                    print(f"   q3: {q3:.4f}")
                    print(f"   q4: {q4:.4f}")
                else:
                    print("   ✗ Failed to read position")
                last_r_press = current_time
                time.sleep(0.1)

            # Position controls
            if keyboard.is_pressed('w'):
                y += pos_speed
                moved = True
                print(f"→ Y: {y:.1f}", end='\r')

            if keyboard.is_pressed('s'):
                y -= pos_speed
                moved = True
                print(f"← Y: {y:.1f}", end='\r')

            if keyboard.is_pressed('a'):
                x -= pos_speed
                moved = True
                print(f"← X: {x:.1f}", end='\r')

            if keyboard.is_pressed('d'):
                x += pos_speed
                moved = True
                print(f"→ X: {x:.1f}", end='\r')

            if keyboard.is_pressed('q'):
                z += pos_speed
                moved = True
                print(f"↑ Z: {z:.1f}", end='\r')

            if keyboard.is_pressed('e'):
                z -= pos_speed
                moved = True
                print(f"↓ Z: {z:.1f}", end='\r')

            # Orientation controls
            if keyboard.is_pressed('i'):
                q_new = rotate_quaternion((q1, q2, q3, q4), (1, 0, 0), rot_speed)
                q1, q2, q3, q4 = q_new
                moved = True
                print(f"⟲ RX+", end='\r')

            if keyboard.is_pressed('k'):
                q_new = rotate_quaternion((q1, q2, q3, q4), (1, 0, 0), -rot_speed)
                q1, q2, q3, q4 = q_new
                moved = True
                print(f"⟳ RX-", end='\r')

            if keyboard.is_pressed('j'):
                q_new = rotate_quaternion((q1, q2, q3, q4), (0, 1, 0), rot_speed)
                q1, q2, q3, q4 = q_new
                moved = True
                print(f"⟲ RY+", end='\r')

            if keyboard.is_pressed('l'):
                q_new = rotate_quaternion((q1, q2, q3, q4), (0, 1, 0), -rot_speed)
                q1, q2, q3, q4 = q_new
                moved = True
                print(f"⟳ RY-", end='\r')

            if keyboard.is_pressed('u'):
                q_new = rotate_quaternion((q1, q2, q3, q4), (0, 0, 1), rot_speed)
                q1, q2, q3, q4 = q_new
                moved = True
                print(f"⟲ RZ+", end='\r')

            if keyboard.is_pressed('o'):
                q_new = rotate_quaternion((q1, q2, q3, q4), (0, 0, 1), -rot_speed)
                q1, q2, q3, q4 = q_new
                moved = True
                print(f"⟳ RZ-", end='\r')

            # Speed adjustment
            if keyboard.is_pressed('+') or keyboard.is_pressed('='):
                pos_speed = min(pos_speed + 0.5, 20.0)
                rot_speed = min(rot_speed + 0.5, 10.0)
                print(f"Speed: {pos_speed}mm/update, {rot_speed}°/update", end='\r')
                time.sleep(0.2)

            if keyboard.is_pressed('-') or keyboard.is_pressed('_'):
                pos_speed = max(pos_speed - 0.5, 0.5)
                rot_speed = max(rot_speed - 0.5, 0.5)
                print(f"Speed: {pos_speed}mm/update, {rot_speed}°/update", end='\r')
                time.sleep(0.2)

            # Send update to robot if any key was pressed
            if moved:
                if arm == "right":
                    set_cartesian_target(x, y, z, q1, q2, q3, q4, verbose=False)
                else:
                    set_cartesian_target_left(x, y, z, q1, q2, q3, q4, verbose=False)

            # Wait for next update
            time.sleep(update_rate)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
    finally:
        cv2.destroyAllWindows()
        print(f"\nFinal position: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
        print(f"Snapshots saved: {snapshot_count}")
        print("Streaming mode ended.")


# -----------------------------
# Shutdown handling
# -----------------------------
def shutdown_handler(sig, frame):
    print("\nStopping robot...")
    hold_rapid()
    SESSION.close()
    sys.exit(0)


# -----------------------------
# Main console loop
# -----------------------------
def main():
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    start_rapid()  # Start RAPID program

    print("\n" + "=" * 60)
    print("ABB YuMi CONTROL - BOTH ARMS SUPPORTED")
    print("=" * 60)
    print("\nCommands:")
    print("  Speed Control:")
    print("    - Type a number (0-100) to set speed")
    print("\n  Position Reading:")
    print("    - 'pos' or 'right' - Read right arm joint positions")
    print("    - 'cart' or 'rightcart' - Read right arm Cartesian position")
    print("    - 'left' - Read left arm joint positions")
    print("    - 'leftcart' - Read left arm Cartesian position")
    print("    - 'all' - Read both arms joint positions")
    print("    - 'allcart' - Read both arms Cartesian positions")
    print("    - 'units' - List available mechanical units")
    print("\n  Movement Commands:")
    print("    - 'move X Y Z' - Move right arm (e.g., 'move 100 200 150')")
    print("    - 'moveleft X Y Z' - Move left arm")
    print("\n  Interactive Control Modes:")
    print("    - 'jog' - Keyboard jog mode for RIGHT arm (step-by-step)")
    print("    - 'jogleft' - Keyboard jog mode for LEFT arm (step-by-step)")
    print("    - 'stream' - Streaming mode for RIGHT arm (hold keys) ⭐")
    print("    - 'streamleft' - Streaming mode for LEFT arm (hold keys) ⭐")
    print("\n  Exit:")
    print("    - Press Ctrl+C to stop and exit")
    print("\n" + "=" * 60 + "\n")

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            parts = user_input.lower().split()
            command = parts[0]

            if command in ["pos", "right"]:
                get_joint_positions("ROB_R")
            elif command in ["cart", "rightcart"]:
                get_cartesian_position("ROB_R")
            elif command == "left":
                get_joint_positions("ROB_L")
            elif command == "leftcart":
                get_cartesian_position("ROB_L")
            elif command == "all":
                get_all_positions()
            elif command == "allcart":
                get_all_cartesian()
            elif command == "units":
                list_mechunits()
            elif command == "jog":
                keyboard_jog(arm="right")
            elif command == "jogleft":
                keyboard_jog(arm="left")
            elif command == "stream":
                streaming_jog(arm="right")
            elif command == "streamleft":
                streaming_jog(arm="left")
            elif command == "move" and len(parts) == 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                move_to(x, y, z, arm="right")
            elif command == "moveleft" and len(parts) == 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                set_cartesian_target_left(x, y, z)
                print("Waiting for move to complete...")
                time.sleep(0.5)
                print("✓ Move command sent")
            else:
                try:
                    speed = int(command)
                    set_speed(speed)
                except ValueError:
                    print("Invalid command. Type a command from the list above.")

        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
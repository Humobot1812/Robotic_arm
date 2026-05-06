import tkinter as tk
from tkinter import ttk
import numpy as np
import serial
import serial.tools.list_ports
import time

# ================= ARM DIMENSIONS =================
L0 = 12.5
L1 = 18.5
L2 = 14.5

# ================= FK =================
def forward_kinematics(s0, s1, s2, s3, gripper):

    t0 = np.radians(s0)
    t1 = np.radians(30 + s1)
    t2 = np.radians(120 - s2)
    t3 = np.radians(90 - s3)

    L3 = 14 + (gripper / 180) * 3

    r = L1*np.cos(t1) + L2*np.cos(t1+t2)
    z = L0 + L1*np.sin(t1) + L2*np.sin(t1+t2)

    x = r*np.cos(t0)
    y = r*np.sin(t0)

    wrist_angle = t1 + t2 + t3

    x_end = x + L3*np.cos(wrist_angle)*np.cos(t0)
    y_end = y + L3*np.cos(wrist_angle)*np.sin(t0)
    z_end = z + L3*np.sin(wrist_angle)

    return x_end, y_end, z_end


# ================= IK =================
def inverse_kinematics(x, y, z):

    theta0 = np.degrees(np.arctan2(y, x))

    r = np.sqrt(x**2 + y**2)
    z = z - L0

    D = (r**2 + z**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = np.clip(D, -1, 1)

    theta2 = np.arccos(D)

    theta1 = np.arctan2(z, r) - np.arctan2(L2*np.sin(theta2), L1 + L2*np.cos(theta2))

    theta3 = -(theta1 + theta2)

    s0 = theta0
    s1 = np.degrees(theta1) - 30
    s2 = 120 - np.degrees(theta2)
    s3 = 90 - np.degrees(theta3)

    return s0, s1, s2, s3


# ================= SEGMENT DISTANCE =================
def segment_distance(p1, p2, p3, p4):

    def dot(a,b): return a[0]*b[0] + a[1]*b[1]
    def sub(a,b): return (a[0]-b[0], a[1]-b[1])
    def norm(a): return np.sqrt(dot(a,a))

    u = sub(p2,p1)
    v = sub(p4,p3)
    w = sub(p1,p3)

    a = dot(u,u)
    b = dot(u,v)
    c = dot(v,v)
    d = dot(u,w)
    e = dot(v,w)

    D = a*c - b*b

    if D < 1e-8:
        sc = 0
        tc = 0
    else:
        sc = (b*e - c*d) / D
        tc = (a*e - b*d) / D

    sc = max(0, min(1, sc))
    tc = max(0, min(1, tc))

    dp = (
        w[0] + sc*u[0] - tc*v[0],
        w[1] + sc*u[1] - tc*v[1]
    )

    return norm(dp)


# ================= COLLISION =================
def check_collision(s0, s1, s2, s3):

    t1 = np.radians(30 + s1)
    t2 = np.radians(120 - s2)
    t3 = np.radians(90 - s3)

    J0 = (0, L0)

    x1 = L1*np.cos(t1)
    z1 = L0 + L1*np.sin(t1)
    J1 = (x1, z1)

    x2 = x1 + L2*np.cos(t1+t2)
    z2 = z1 + L2*np.sin(t1+t2)
    J2 = (x2, z2)

    L3 = 14
    wrist_angle = t1 + t2 + t3

    x3 = x2 + L3*np.cos(wrist_angle)
    z3 = z2 + L3*np.sin(wrist_angle)
    J3 = (x3, z3)

    MIN_DIST = 3.5

    if segment_distance(J0, J1, J2, J3) < MIN_DIST:
        return True

    return False


# ================= UI =================
class ArmUI:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ Collision-Safe Arm + IK")
        self.root.configure(bg="#0d0d0d")

        self.ser = None
        self.default_pose = [0, 0, 0, 0, 90]

        # ===== CONNECTION =====
        conn = tk.Frame(root, bg="#0d0d0d")
        conn.pack(fill="x")

        tk.Label(conn, text="PORT", fg="cyan", bg="#0d0d0d").pack(side="left")

        self.port_var = tk.StringVar()
        self.port_menu = ttk.Combobox(conn, textvariable=self.port_var, width=10)
        self.port_menu.pack(side="left", padx=5)

        tk.Button(conn, text="REFRESH", command=self.refresh_ports).pack(side="left")
        tk.Button(conn, text="CONNECT", command=self.toggle_connection).pack(side="left")

        self.status = tk.Label(conn, text="● DISCONNECTED", fg="red", bg="#0d0d0d")
        self.status.pack(side="left", padx=10)

        # ===== CANVAS =====
        self.w = 500
        self.h = 450
        self.canvas = tk.Canvas(root, width=self.w, height=self.h,
                                bg="#050505", highlightthickness=0)
        self.canvas.pack()

        # ===== MODE SWITCH =====
        mode_frame = tk.Frame(root, bg="#0d0d0d")
        mode_frame.pack(pady=5)

        self.mode = "FK"

        tk.Button(mode_frame, text="FK MODE", bg="cyan",
                command=self.set_fk_mode).pack(side="left", padx=5)

        tk.Button(mode_frame, text="IK MODE", bg="magenta",
                command=self.set_ik_mode).pack(side="left", padx=5)

        # ===== SLIDERS =====
        self.base = self.slider("BASE")
        self.shoulder = self.slider("SHOULDER")
        self.elbow = self.slider("ELBOW")
        self.wrist = self.slider("WRIST")
        self.gripper = self.slider("GRIPPER")

        # ===== IK SLIDERS =====
        # ===== IK FRAME (ADDED WRAPPER) =====
        
        self.ik_frame = tk.Frame(root, bg="#0d0d0d")
        self.ik_frame.pack()

        tk.Label(self.ik_frame, text="--- IK CONTROL ---", fg="magenta", bg="#0d0d0d").pack()

        self.tx = self.slider("X")
        self.ty = self.slider("Y")
        self.tz = self.slider("Z")

        self.tx.config(from_=-30, to=30)
        self.ty.config(from_=-30, to=30)
        self.tz.config(from_=0, to=40)

        tk.Button(self.ik_frame, text="MOVE IK", bg="purple", fg="white",
                command=self.move_ik).pack(pady=5)

        self.label = tk.Label(root, fg="cyan", bg="#0d0d0d")
        self.label.pack()

        # ===== BUTTONS =====
        btn = tk.Frame(root, bg="#0d0d0d")
        btn.pack()

        tk.Button(btn, text="EXECUTE", command=self.execute).pack(side="left")
        tk.Button(btn, text="RESET", command=self.set_default).pack(side="left")

        self.refresh_ports()
        self.set_default()

    def slider(self, name):
        f = tk.Frame(self.root, bg="#0d0d0d")
        f.pack(fill="x")

        tk.Label(f, text=name, fg="cyan", bg="#0d0d0d", width=10).pack(side="left")

        s = tk.Scale(f, from_=0, to=180, orient='horizontal',
                     bg="#0d0d0d", fg="cyan",
                     command=self.update)
        s.pack(fill="x")
        return s

    def move_ik(self):
        
        if self.mode != "IK":
            return
        
        x = self.tx.get()
        y = self.ty.get()
        z = self.tz.get()

        try:
            s0, s1, s2, s3 = inverse_kinematics(x, y, z)

            s1 = max(0, min(180, s1))
            s2 = max(0, min(180, s2))
            s3 = max(0, min(180, s3))

            if check_collision(s0, s1, s2, s3):
                self.status.config(text="⚠ IK COLLISION", fg="orange")
                return

            self.base.set(s0)
            self.shoulder.set(s1)
            self.elbow.set(s2)
            self.wrist.set(s3)

            self.update()

        except:
            self.status.config(text="❌ IK FAILED", fg="red")
            
    def set_fk_mode(self):
        self.mode = "FK"
        self.ik_frame.pack_forget()
        self.status.config(text="● FK MODE", fg="cyan")


    def set_ik_mode(self):
        self.mode = "IK"
        self.ik_frame.pack()
        self.status.config(text="● IK MODE", fg="magenta")

    def refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        self.port_menu['values'] = [p.device for p in ports]
        if ports:
            self.port_var.set(ports[0].device)

    def toggle_connection(self):
        try:
            self.ser = serial.Serial(self.port_var.get(), 115200)
            time.sleep(2)
            self.status.config(text="● CONNECTED", fg="lime")
        except:
            self.status.config(text="● FAILED", fg="red")

    def draw(self):
        self.canvas.delete("all")

        max_reach = L0 + L1 + L2 + 17
        scale = (self.h * 0.75) / max_reach

        ox = self.w // 2
        oy = int(self.h * 0.9)

        s0 = self.base.get()
        s1 = self.shoulder.get()
        s2 = self.elbow.get()
        s3 = self.wrist.get()
        g  = self.gripper.get()

        t0 = np.radians(s0)
        t1 = np.radians(30 + s1)
        t2 = np.radians(120 - s2)
        t3 = np.radians(90 - s3)

        x_base, y_base = ox, oy - L0*scale

        x1 = x_base + L1*np.cos(t1)*np.cos(t0)*scale
        y1 = y_base - L1*np.sin(t1)*scale

        x2 = x1 + L2*np.cos(t1+t2)*np.cos(t0)*scale
        y2 = y1 - L2*np.sin(t1+t2)*scale

        L3 = 14 + (g/180)*3
        wrist_angle = t1 + t2 + t3

        x3 = x2 + L3*np.cos(wrist_angle)*np.cos(t0)*scale
        y3 = y2 - L3*np.sin(wrist_angle)*scale

        self.canvas.create_line(ox, oy, x_base, y_base, fill="white", width=4)
        self.canvas.create_line(x_base, y_base, x1, y1, fill="cyan", width=4)
        self.canvas.create_line(x1, y1, x2, y2, fill="lime", width=4)
        self.canvas.create_line(x2, y2, x3, y3, fill="yellow", width=3)

        self.canvas.create_oval(x3-4, y3-4, x3+4, y3+4, fill="white")

       
    
    def update(self, e=None):

        s0 = self.base.get()
        s1 = self.shoulder.get()
        s2 = self.elbow.get()
        s3 = self.wrist.get()

        collision = check_collision(s0, s1, s2, s3)

        if collision:
            self.status.config(text="⚠ COLLISION", fg="orange")
        else:
            if self.mode == "FK":
                self.status.config(text="● FK MODE", fg="cyan")
            else:
                self.status.config(text="● IK MODE", fg="magenta")

        x, y, z = forward_kinematics(
            s0, s1, s2, s3, self.gripper.get()
        )

        self.label.config(text=f"TIP → X:{x:.1f} Y:{y:.1f} Z:{z:.1f}")
        self.draw()

    def execute(self):
        if self.ser:
            data = f"{self.base.get()},{self.shoulder.get()},{self.elbow.get()},{self.wrist.get()},{self.gripper.get()}\n"
            self.ser.write(data.encode())

    def set_default(self):
        vals = self.default_pose
        self.base.set(vals[0])
        self.shoulder.set(vals[1])
        self.elbow.set(vals[2])
        self.wrist.set(vals[3])
        self.gripper.set(vals[4])
        self.update()


# ================= MAIN =================
root = tk.Tk()
app = ArmUI(root)
root.mainloop()
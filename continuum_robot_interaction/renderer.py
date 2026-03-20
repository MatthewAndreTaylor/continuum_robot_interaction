import numpy as np
from PIL import Image
import pickle
from scipy import interpolate
import glfw
from OpenGL.GL import (
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor4f,
    glColorMaterial,
    glDisable,
    glEnable,
    glEnd,
    glLightfv,
    glLoadIdentity,
    glMaterialf,
    glMatrixMode,
    glNormal3f,
    glPixelStorei,
    glReadPixels,
    glVertex3f,
    glViewport,
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHT1,
    GL_LIGHTING,
    GL_MODELVIEW,
    GL_NORMALIZE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PACK_ALIGNMENT,
    GL_POSITION,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGB,
    GL_SHININESS,
    GL_SPECULAR,
    GL_SRC_ALPHA,
    GL_UNSIGNED_BYTE,
)
from OpenGL.GLU import gluLookAt, gluPerspective
from dataclasses import dataclass

# COLORS
# ROD1_COLOR = np.array([0.45, 0.39, 1.0])   # blue-purple  - upper section
# ROD2_COLOR = np.array([0.20, 0.80, 0.50])  # teal-green   - lower section

ROD1_COLOR = np.array([0.25, 0.30, 1.00])  # deeper blue-violet (upper)
ROD2_COLOR = np.array([0.25, 0.30, 1.00])  # bright blue (lower)
OBSTACLE_COLOR = np.array([1.0, 0.50, 0.10])  # bright orange


@dataclass
class RenderScene:
    headless: bool = False
    width: int = 1000
    height: int = 1000
    fps: float = 60.0


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=0, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def compute_parallel_transport_frames(centerline):
    tangents = np.gradient(centerline, axis=1)
    tangents = normalize_vectors(tangents)

    num_points = centerline.shape[1]
    normals = np.zeros_like(centerline)
    binormals = np.zeros_like(centerline)

    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference, tangents[:, 0])) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    initial_normal = np.cross(tangents[:, 0], reference)
    n = np.linalg.norm(initial_normal)
    normals[:, 0] = initial_normal / (n if n > 1e-12 else 1.0)
    binormals[:, 0] = np.cross(tangents[:, 0], normals[:, 0])
    binormals[:, 0] /= max(np.linalg.norm(binormals[:, 0]), 1e-12)

    for idx in range(1, num_points):
        prev_normal = normals[:, idx - 1]
        tangent = tangents[:, idx]
        normal = prev_normal - np.dot(prev_normal, tangent) * tangent
        normal_norm = np.linalg.norm(normal)

        if normal_norm < 1e-12:
            reference = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(reference, tangent)) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            normal = np.cross(tangent, reference)
            normal_norm = np.linalg.norm(normal)

        normals[:, idx] = normal / max(normal_norm, 1e-12)
        binormals[:, idx] = np.cross(tangent, normals[:, idx])
        binormals[:, idx] /= max(np.linalg.norm(binormals[:, idx]), 1e-12)

    return tangents, normals, binormals


def build_tube_mesh(centerline, radii, sides=12):
    """Build a tube surface mesh around a centerline."""
    _, normals, binormals = compute_parallel_transport_frames(centerline)
    theta = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos_theta = np.cos(theta)[:, None]
    sin_theta = np.sin(theta)[:, None]

    offsets = (
        normals[None, :, :] * cos_theta[:, None, :]
        + binormals[None, :, :] * sin_theta[:, None, :]
    )
    mesh = centerline[None, :, :] + offsets * radii[None, None, :]
    return mesh[:, 0, :], mesh[:, 1, :], mesh[:, 2, :]


def build_obstacle_mesh(start, direction, length, radius):
    """Return the static tube mesh for the fixed cylinder obstacle."""
    t = np.linspace(0.0, length, 20)
    centerline = start[:, None] + direction[:, None] * t[None, :]
    radii = np.full(20, radius)
    return build_tube_mesh(centerline, radii, sides=16)


def _mesh_to_quads(x, y, z):
    """Convert tube grid mesh into a list of quads for OpenGL immediate rendering."""
    points = np.stack([x, y, z], axis=-1)
    ring_count, seg_count, _ = points.shape
    quads = []
    for i in range(ring_count):
        ni = (i + 1) % ring_count
        for j in range(seg_count - 1):
            p00 = points[i, j]
            p10 = points[ni, j]
            p11 = points[ni, j + 1]
            p01 = points[i, j + 1]
            quads.append((p00, p10, p11, p01))
    return quads


def _quad_normal(p0, p1, p3):
    normal = np.cross(p1 - p0, p3 - p0)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return np.array([0.0, 1.0, 0.0])
    return normal / norm


def _draw_quads(quads, color, alpha=1.0):
    glColor4f(color[0], color[1], color[2], alpha)
    glBegin(GL_QUADS)
    for p0, p1, p2, p3 in quads:
        n = _quad_normal(p0, p1, p3)
        glNormal3f(n[0], n[1], n[2])
        glVertex3f(p0[0], p0[1], p0[2])
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p2[0], p2[1], p2[2])
        glVertex3f(p3[0], p3[1], p3[2])
    glEnd()


class GLFWViewer:
    def __init__(
        self,
        scene: RenderScene,
    ):
        self.scene = scene
        self.window = None

        # TODO: camera parameters configurable through scene
        self.yaw = np.deg2rad(35.0)
        self.pitch = np.deg2rad(30.0)
        self.distance = 1.6

        # Centre on the midpoint between the arm pivot (y=0) and tip (y=-0.5),
        # with z offset halfway toward the cylinder (z=0.2) so both are in frame.
        self.target = np.array([0.0, -0.25, 0.1])
        self.dragging = False
        self.last_cursor = None
        self.playing = True
        self.frame_idx = 0
        self.last_frame_time = 0.0

    def _on_cursor_pos(self, _window, xpos, ypos):
        if not self.dragging or self.last_cursor is None:
            self.last_cursor = (xpos, ypos)
            return

        dx = xpos - self.last_cursor[0]
        dy = ypos - self.last_cursor[1]
        self.last_cursor = (xpos, ypos)
        self.yaw -= dx * 0.005
        self.pitch -= dy * 0.005
        self.pitch = np.clip(self.pitch, -1.4, 1.4)

    def _on_mouse_button(self, window, button, action, _mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == glfw.PRESS
            if not self.dragging:
                self.last_cursor = None
            else:
                self.last_cursor = glfw.get_cursor_pos(window)

    def _on_scroll(self, _window, _xoff, yoff):
        self.distance *= np.exp(-0.12 * yoff)
        self.distance = np.clip(self.distance, 0.3, 5.0)

    def _on_key(self, window, key, _scancode, action, _mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        elif key == glfw.KEY_SPACE:
            self.playing = not self.playing
        elif key == glfw.KEY_R:
            self.frame_idx = 0
            self.playing = True

    def _set_camera(self):
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        eye = self.target + self.distance * np.array([cp * cy, sp, cp * sy])

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            eye[0],
            eye[1],
            eye[2],
            self.target[0],
            self.target[1],
            self.target[2],
            0.0,
            1.0,
            0.0,
        )

    def _configure_gl(self):
        glClearColor(0.97, 0.98, 1.00, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)

        # Slightly glossy response helps visualize curved tube geometry
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 24.0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.18, 0.18, 0.18, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.95, 0.95, 0.90, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.70, 0.70, 0.65, 1.0))
        glLightfv(GL_LIGHT1, GL_AMBIENT, (0.05, 0.05, 0.07, 1.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.35, 0.42, 0.50, 1.0))
        glLightfv(GL_LIGHT1, GL_SPECULAR, (0.20, 0.25, 0.30, 1.0))

    def _update_lights(self):
        # Transform light positions to current model-view matrix.
        glLightfv(GL_LIGHT0, GL_POSITION, (1.2, 1.5, 1.4, 1.0))
        glLightfv(GL_LIGHT1, GL_POSITION, (-1.6, 0.8, -0.9, 1.0))

    def _set_projection(self):
        aspect = max(self.width / max(self.height, 1), 1e-6)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, aspect, 0.01, 30.0)

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE if self.scene.headless else glfw.TRUE)
        self.window = glfw.create_window(self.scene.width, self.scene.height, "Viewer", None, None)

        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)
        self._configure_gl()

    def _draw_frame(self, xs1, xs2, rod_radius, idx, obstacle_quads):
        self.width, self.height = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, self.width, self.height)
        self._set_projection()
        self._set_camera()
        self._update_lights()

        x1, y1, z1 = build_tube_mesh(
            xs1[idx], np.full(xs1.shape[2], rod_radius), sides=12
        )
        x2, y2, z2 = build_tube_mesh(
            xs2[idx], np.full(xs2.shape[2], rod_radius), sides=12
        )

        rod1_quads = _mesh_to_quads(x1, y1, z1)
        rod2_quads = _mesh_to_quads(x2, y2, z2)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        _draw_quads(obstacle_quads, OBSTACLE_COLOR, alpha=0.75)
        _draw_quads(rod1_quads, ROD1_COLOR)
        _draw_quads(rod2_quads, ROD2_COLOR)

    def save_screenshot(self, image_path):
        width, height = glfw.get_framebuffer_size(self.window)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        buffer = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
        image = np.flipud(image)
        Image.fromarray(image).save(image_path)

    def run(self, xs1, xs2, rod_radius, obs_details, fps, screenshot_path=None):
        self._init_window()

        try:
            x_obs, y_obs, z_obs = build_obstacle_mesh(*obs_details)
            obstacle_quads = _mesh_to_quads(x_obs, y_obs, z_obs)
            frame_count = xs1.shape[0]

            if self.scene.headless:
                self._draw_frame(xs1, xs2, rod_radius, frame_count - 1, obstacle_quads)
                if screenshot_path is not None:
                    self.save_screenshot(screenshot_path)
                return

            while not glfw.window_should_close(self.window):
                glfw.poll_events()

                now = glfw.get_time()
                if self.playing and (now - self.last_frame_time) >= 1.0 / fps:
                    if self.frame_idx < frame_count - 1:
                        self.frame_idx += 1
                    else:
                        self.playing = False
                    self.last_frame_time = now

                self._draw_frame(
                    xs1,
                    xs2,
                    rod_radius,
                    self.frame_idx,
                    obstacle_quads,
                )
                glfw.swap_buffers(self.window)
        finally:
            if self.window is not None:
                glfw.destroy_window(self.window)
                self.window = None
            glfw.terminate()


def render_sim(data_path, scene = RenderScene()):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        
    fps = scene.fps
    # Currently rods and times are hard coded fields
    times1 = np.array(data["rod1"]["time"])
    xs1 = np.array(data["rod1"]["position"])
    times2 = np.array(data["rod2"]["time"])
    xs2 = np.array(data["rod2"]["position"])
    rod_radius = data.get("rod_radius", 0.025)

    # obstacle parameters read optionally
    # defaults match sim default
    obs_start = np.array(data.get("obstacle_start", [0, -0.70, 0.1]))
    obs_direction = np.array(data.get("obstacle_direction", [0.0, 1.0, 0.0]))
    obs_length = data.get("obstacle_length", 1.4)
    obstacle_radius = data.get("obstacle_radius", 0.03)
    obs_details = (obs_start, obs_direction, obs_length, obstacle_radius)

    runtime = times1.max()
    total_frame = int(runtime * fps)
    t_uniform = np.linspace(0, runtime, total_frame)
    xs1 = interpolate.interp1d(times1, xs1, axis=0)(t_uniform)
    xs2 = interpolate.interp1d(times2, xs2, axis=0)(t_uniform)
    viewer = GLFWViewer(scene)
    viewer.run(xs1, xs2, rod_radius, obs_details, fps=fps)

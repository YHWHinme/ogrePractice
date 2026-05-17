from pathlib import Path

import cv2
import glfw
import numpy as np
import pywavefront
from OpenGL.GL import *

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MARKER_LENGTH = 0.05
MODEL_SCALE = 0.5
NEAR = 0.01
FAR = 100.0
MODEL_PATH = str(Path(__file__).parent / "models" / "ShowcaseModel.obj")

FX = FRAME_WIDTH
FY = FRAME_WIDTH
CX = FRAME_WIDTH / 2.0
CY = FRAME_HEIGHT / 2.0

CAMERA_MATRIX = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

DIST_COEFFS = np.zeros(4, dtype=np.float64)

VERTEX_FORMATS = {
    "V3F": GL_V3F,
    "C3F_V3F": GL_C3F_V3F,
    "N3F_V3F": GL_N3F_V3F,
    "T2F_V3F": GL_T2F_V3F,
    "T2F_C3F_V3F": GL_T2F_C3F_V3F,
    "T2F_N3F_V3F": GL_T2F_N3F_V3F,
}


def projection_matrix(k, w, h, near, far):
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    left = (cx - w) * near / fx
    right = cx * near / fx
    btm = (cy - h) * near / fy
    top = cy * near / fy
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] = 2 * near / (right - left)
    p[1, 1] = 2 * near / (top - btm)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + btm) / (top - btm)
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -1
    p[3, 2] = -2 * far * near / (far - near)
    return p.T.flatten()


def modelview_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = t
    flip = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], np.float32
    )
    return (flip @ np.linalg.inv(pose)).T.flatten()


def draw_material(material):
    """Draw a single material using interleaved vertex arrays"""
    print(
        f"draw_material: vertices={len(material.vertices) if material.vertices else 0}, format={material.vertex_format}"
    )

    if not material.vertices:
        print("  -> no vertices, returning")
        return

    vertex_size = material.vertex_size
    triangle_count = int(len(material.vertices) / vertex_size)
    print(f"  -> vertex_size={vertex_size}, triangle_count={triangle_count}")

    try:
        gl_floats = (GLfloat * len(material.vertices))(*material.vertices)
    except Exception as e:
        print(f"  -> ERROR: {e}")
        return

    vertex_format = VERTEX_FORMATS.get(material.vertex_format)
    print(f"  -> vertex_format={vertex_format} (from '{material.vertex_format}')")
    if not vertex_format:
        print("  -> ERROR: unknown format")
        return

    print("  -> calling glInterleavedArrays and glDrawArrays")
    glInterleavedArrays(vertex_format, 0, gl_floats)
    glDrawArrays(GL_TRIANGLES, 0, triangle_count)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("error: cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    try:
        scene = pywavefront.Wavefront(MODEL_PATH, collect_faces=True)
        materials = []
        for name, material in scene.materials.items():
            if material.vertices:
                materials.append(material)
        print(f"loaded {len(materials)} material(s) from {MODEL_PATH}")
    except Exception as e:
        print(f"error loading model: {e}")
        materials = []

    dict_ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(dict_, cv2.aruco.DetectorParameters())

    if not glfw.init():
        return
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    win = glfw.create_window(FRAME_WIDTH, FRAME_HEIGHT, "ArUco AR", None, None)
    if not win:
        glfw.terminate()
        return
    glfw.make_context_current(win)

    proj = projection_matrix(CAMERA_MATRIX, FRAME_WIDTH, FRAME_HEIGHT, NEAR, FAR)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.3, 0.3, 0.3, 1])

    while not glfw.window_should_close(win):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        print(ids)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            rgb,
        )

        glViewport(0, 0, FRAME_WIDTH, FRAME_HEIGHT)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_TEXTURE_2D)
        glColor3f(1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1)
        glVertex2f(-1, -1)
        glTexCoord2f(1, 1)
        glVertex2f(1, -1)
        glTexCoord2f(1, 0)
        glVertex2f(1, 1)
        glTexCoord2f(0, 0)
        glVertex2f(-1, 1)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        # Validation: draw a red rectangle at the top
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glColor3f(1, 0, 0)
        glBegin(GL_QUADS)
        glVertex2f(-0.2, 0.7)
        glVertex2f(0.2, 0.7)
        glVertex2f(0.2, 0.9)
        glVertex2f(-0.2, 0.9)
        glEnd()

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, CAMERA_MATRIX, DIST_COEFFS
            )

            glMatrixMode(GL_PROJECTION)
            glLoadMatrixf(proj)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)

            for i in range(len(ids)):
                mv = modelview_matrix(rvecs[i][0], tvecs[i][0])
                glLoadMatrixf(mv)
                glScalef(MODEL_SCALE, MODEL_SCALE, MODEL_SCALE)
                for material in materials:
                    draw_material(material)

            glDisable(GL_DEPTH_TEST)
            glDisable(GL_LIGHTING)

        glfw.swap_buffers(win)
        glfw.poll_events()

    cap.release()
    glfw.terminate()


if __name__ == "__main__":
    main()

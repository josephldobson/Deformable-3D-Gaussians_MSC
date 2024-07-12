import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mayavi import mlab
import numpy as np
import sys

def load_and_animate_deformation():
    x_full = np.load('x_full.npy')
    y_full = np.load('y_full.npy')
    z_full = np.load('z_full.npy')
    u_full = np.load('u_full.npy')
    v_full = np.load('v_full.npy')
    w_full = np.load('w_full.npy')
    s = np.load('s.npy')

    num_frames = x_full.shape[0]
    
    xs, ys, zs = x_full[0], y_full[0], z_full[0]
    us, vs, ws = u_full[0], v_full[0], w_full[0]
    
    points = mlab.points3d(xs, ys, zs, s, scale_factor=0.01, color=(1, 0.1, 0))
    vectors = mlab.quiver3d(xs, ys, zs, us, vs, ws, line_width=0.5, scale_factor=0.05, color=(0, 1, 0))


    mlab.xlabel('X-axis')
    mlab.ylabel('Y-axis')
    mlab.zlabel('Z-axis')
    
    @mlab.animate(delay=600)
    def animate():
        for frame in range(num_frames):
            points.mlab_source.set(x=x_full[frame], y=y_full[frame], z=z_full[frame])
            vectors.mlab_source.set(x=x_full[frame], y=y_full[frame], z=z_full[frame], u=u_full[frame], v=v_full[frame], w=w_full[frame])
            print(frame)
            yield
    
    # Start animation
    animate()
    mlab.show()

# Example usage:
load_and_animate_deformation()
